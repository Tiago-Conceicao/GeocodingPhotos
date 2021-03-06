import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.applications import densenet
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.color import rgb2lab
from data_utils import geodistance_tensorflow, get_results, latlon2healpix, healpix2latlon, cube, get_values_from_matrix
from clr import CyclicLR
from efficientnet import EfficientNetB3, EfficientNetB4, EfficientNetB5
import tensorflow as tf
import numpy as np
import random as rn
import os
import math
import healpy


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))


_MODEL_WEIGHTS_FINAL_NAME = '/home/tconceicao/weights/effnet_ny_old.h5'
_TRAIN_TEST_SPLIT_SEED = 2
LC_PATH = "geo_files/historic_landcover_hd_1900.asc"
_PATH = '/home/tconceicao/old_sf_300/'
resolution = math.pow(2, 12)
image_size = 300
batch_size = 8

def get_images(path):
    images_list = os.listdir(path) #list of all images
    images = []
    coordinates = []
    for line in images_list:
        images.append(line)
        entry = os.path.splitext(line)[0].split(",") #filename without the extension
        coordinates.append((entry[1].rstrip(), entry[2]))
    #return images[:sample], coordinates[:sample]
    return images, coordinates

def generate_arrays_from_file(X, Y1, Y2, Y3, codes_matrix, terrain_matrix, batchsize):
    while 1:
        line = -1
        new_X = np.zeros((batchsize, image_size, image_size, 3))
        new_Y1 = np.zeros((batchsize, 2))
        new_Y2 = np.zeros((batchsize, len(Y2[0])))
        new_Y3 = np.zeros((batchsize, len(Y3[0])))

        count = 0
        for entry in X:
            if count < batchsize:
                line+=1
                x_b = load_img(_PATH+entry)
                x_b = img_to_array(x_b)
                if (rn.randint(0, 1) == 1):
                    x_b = np.fliplr(x_b)
                x_b = np.expand_dims(x_b, axis=0)
                x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.
                x_b = rgb2lab(x_b)

                y = [float(Y1[line][0]), float(Y1[line][1])]

                new_X[count,:] = x_b
                new_Y1[count,:] = np.array(y)
                new_Y2[count,:] = np.array([Y2[line]])
                new_Y3[count, :] = np.array([Y3[line]])

                count+=1
            else:
                yield [new_X, np.asarray(codes_matrix * len(new_X)), np.asarray(terrain_matrix * len(new_X))]
                count = 0
                new_X = np.zeros((batchsize, image_size, image_size, 3))
                new_Y1 = np.zeros((batchsize, 2))
                new_Y2 = np.zeros((batchsize, len(Y2[0])))
                new_Y3 = np.zeros((batchsize, len(Y3[0])))
        if(np.count_nonzero(new_X) != 0):
            yield [new_X, np.asarray(codes_matrix * len(new_X)), np.asarray(terrain_matrix * len(new_X)) ]


#Load training images
images, coordinates = get_images(_PATH)
X_train, X_test, Y_train, Y_test = train_test_split(images, coordinates,
                                    test_size=0.20, random_state = _TRAIN_TEST_SPLIT_SEED)

print('IMAGES LOADED')

####mfc#####
encoder = LabelEncoder()
region_codes = []
for c in (coordinates):
    region = latlon2healpix(float(c[0]), float(c[1]), resolution)
    region_codes.append(region)
classes = to_categorical(encoder.fit_transform(region_codes))

mfc = max(set(region_codes), key=region_codes.count)
mfc_coordinates = healpix2latlon(mfc, resolution)
#############

train_instances_sz = len(Y_train)
test_instances_sz = len(Y_test)
print("Train instances:", train_instances_sz)
print("Test instances:", test_instances_sz)

land_coverage = []
print("Reading land coverage data...")
with open(LC_PATH, 'r') as f:
    c = 0
    for line in f:
        c += 1
        if (c <= 6): continue
        row = np.asarray(line.split(' '))
        land_coverage.append(row.astype(int))


encoder = LabelEncoder()
region_codes = []
terrain_labels = []
print("Building Y labels")
for coordinates in (Y_train + Y_test):
    lat = float(coordinates[0])
    lon = float(coordinates[1])
    region = latlon2healpix(lat, lon, resolution)
    region_codes.append(region)

    aux = get_values_from_matrix(land_coverage, lat, lon)
    #print("Terrain: ", aux)
    terrain_labels.append(aux)

classes = to_categorical(encoder.fit_transform(region_codes))

Y1_train = np.array(Y_train)
Y1_test = np.array(Y_test)

Y2_train = classes[:train_instances_sz]
Y2_test = classes[-test_instances_sz:]

Y3_train = np.array(terrain_labels[:train_instances_sz])
Y3_test = np.array(terrain_labels[-test_instances_sz:])

# construction of a matrix, where each row is the centroid coordinates (x,y,z)
# that corresponds to each different region class
# number of rows is equal to the number of existing region classes
region_list = [i for i in range(Y2_train.shape[1])]
region_classes = encoder.inverse_transform(region_list)
codes_matrix = []
terrain_matrix = [] # matrix where each row is a one hot array of 20 terrain types

for i in range(len(region_classes)):
    [xs, ys, zs] = healpy.pix2vec( int(resolution), region_classes[i] )

    codes_matrix.append([xs, ys, zs])

    lat, lon =  healpix2latlon(region_classes[i], int(resolution))
    aux = get_values_from_matrix(land_coverage, lat, lon)

    terrain_matrix.append(aux)

codes_matrix = [codes_matrix]

terrain_matrix = [terrain_matrix]


######### MODEL ################

efficientNet = EfficientNetB3(input_shape=(image_size,image_size,3), classes=1000, include_top=True, weights='imagenet')

#build model
codes_input_matrix = layers.Input(shape=(Y2_train.shape[1], 3), dtype='float32', name="codes_matrix")
terrain_input_matrix = layers.Input(shape=(Y2_train.shape[1], 20), dtype='float32', name="terrain_matrix")

inp = layers.Input(shape=(image_size, image_size, 3))
x = efficientNet(inp)

auxiliary_output = layers.Dense(Y2_train.shape[1], activation='softmax', name = "auxiliary_output")(x)
cube_prob = layers.Lambda(cube)(auxiliary_output)

main_output = layers.dot([cube_prob, codes_input_matrix], axes=1, name='main_output')
terrain_output = layers.dot([cube_prob, terrain_input_matrix], axes=1, name='terrain_output')

model = models.Model(inputs=[inp, codes_input_matrix, terrain_input_matrix],
                     outputs=[main_output, auxiliary_output, terrain_output])
################################


print("Computing predictions...")
model.load_weights(_MODEL_WEIGHTS_FINAL_NAME)
predictions = model.predict_generator(generate_arrays_from_file(X_test, Y_test, Y2_test, Y3_test, codes_matrix, terrain_matrix, batch_size), steps = test_instances_sz/batch_size)

print("Computing accuracy...\n")
print("Coordinates results:")
get_results(predictions, test_instances_sz, encoder, resolution, Y1_test, codes_flag=0)
'''
print("Centroids results:")
get_results(predictions, len(images), encoder, resolution, Y_train+Y_test, codes_flag=1)
print("Mean results:")
get_results(predictions, len(images), encoder, resolution, Y_train+Y_test, codes_flag=-2)
'''