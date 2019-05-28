import keras
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras import optimizers, backend as K
from sklearn.model_selection import train_test_split #cross_validation -> model_selection
from skimage.color import rgb2lab
import sys
from data_utils import geodistance_theano, normalize_values, get_results, latlon2healpix, healpix2latlon
from model_utils_new import regression_model
from keras.callbacks import ModelCheckpoint
import keras
from clr import CyclicLR
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import random as rn
import os
import math
from keras.utils.np_utils import to_categorical


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

#_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny
_MODEL_FINAL_NAME = '/home/tconceicao/weights/mobilenet_sf_regression.h5'
_MODEL_WEIGHTS_FINAL_NAME = '/home/tconceicao/weights/mobilenet_sf_regression_weights.h5'
#_PATH = '../../../../home/nramanlal/data-nramanlal-lfreixinho/resized_images_sf/'
#_PATH = '/home/tconceicao/resized_images_sf2/'
_PATH = '/home/tconceicao/resized_images_ny2/'
_X_TEST_FILE = 'x_test_regression_sf.txt'
_TRAIN_TEST_SPLIT_SEED = 2
num_per_epoch = 100
batch_size = 36
sample = 15000


def get_images(path):
    images_list = os.listdir(path) #list of all images
    images = []
    coordinates = []
    for line in images_list:
        images.append(line)
        entry = os.path.splitext(line)[0].split(",") #filename without the extension
        coordinates.append((entry[2].rstrip(), entry[1]))
    return images[:sample], coordinates[:sample]
'''
def generate_arrays_from_file(X, Y, batch_size):
    while 1:
        line = -1
        new_X = np.zeros((batch_size, 224, 224, 3))
        new_Y = np.zeros((batch_size, 2))
        count = 0
        for entry in X:
            if count < batch_size:
                line+=1
                x_b = load_img(_PATH+entry)
                x_b = img_to_array(x_b)
                if (rn.randint(0, 1) == 1):
                    x_b = np.fliplr(x_b)
                x_b = np.expand_dims(x_b, axis=0)
                x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.
                x_b = rgb2lab(x_b)

                a = normalize_values(Y[line][0], _BOUNDING_BOX[0], _BOUNDING_BOX[1] )
                b = normalize_values(Y[line][1],_BOUNDING_BOX[2], _BOUNDING_BOX[3])
                y = [(float(a), float(b))]

                new_X[count,:] = x_b
                new_Y[count,:] = np.array(y)
                count+=1
            else:
                yield (new_X, new_Y)
                count = 0
                new_X = np.zeros((batch_size, 224, 224, 3))
                new_Y = np.zeros((batch_size, 2))

        if(np.count_nonzero(new_X) != 0):
                yield (new_X, new_Y)
'''
def generate_arrays_from_file(X, Y, classes_, batchsize):
    while 1:
        line = -1
        new_X = np.zeros((batchsize, 224, 224, 3))
        new_Y2 = np.zeros((batchsize, len(classes_[0])))
        new_Y1 = np.zeros((batchsize, 2))
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

                a = normalize_values(Y[line][0], _BOUNDING_BOX[0], _BOUNDING_BOX[1] )
                b = normalize_values(Y[line][1], _BOUNDING_BOX[2], _BOUNDING_BOX[3])
                y = [(float(a), float(b))]

                z = [classes_[line]]
                new_X[count,:] = x_b
                new_Y1[count,:] = np.array(y)
                new_Y2[count,:] = np.array(z)
                count+=1
            else:
                yield (new_X, [new_Y1,new_Y2])
                count = 0
                new_X = np.zeros((batchsize, 224, 224, 3))
                new_Y1 = np.zeros((batchsize, 2))
                new_Y2 = np.zeros((batchsize, len(classes_[0])))
        if(np.count_nonzero(new_X) != 0):
                yield (new_X, [new_Y1,new_Y2])

#Load training images
training_images, training_coordinates = get_images(_PATH)
X_train, X_test, Y_train, Y_test = train_test_split(training_images, training_coordinates,
                                    test_size=0.20, random_state = _TRAIN_TEST_SPLIT_SEED)
X_train_size = len(X_train)


'''
toWrite = ''
for inst in X_test:
    toWrite += inst
    toWrite += '\n'

file = open(_X_TEST_FILE, 'w')
file.write(str(toWrite))
file.close()
'''

print('IMAGES LOADED')

res = math.pow(4,4)
encoder = LabelEncoder()
region_codes = []
for coordinates in (Y_train+Y_test):
    region = latlon2healpix(float(coordinates[0]), float(coordinates[1]), res)
    region_codes.append(region)
classes = to_categorical(encoder.fit_transform(region_codes))
nclasses = len(classes[0])

Y_train_class = classes[:-len(Y_test)]


model = regression_model(nclasses)
checkpoint = ModelCheckpoint(_MODEL_WEIGHTS_FINAL_NAME, monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True)

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

model.compile(loss={"main_output": geodistance_theano, "auxiliary_output": 'categorical_crossentropy'}, optimizer = opt,
              loss_weights=[0.5, 0.5], metrics={"main_output": geodistance_theano, "auxiliary_output": 'categorical_accuracy'})

earlyStopping=keras.callbacks.EarlyStopping(monitor = 'loss', patience=5, verbose=1)

step = 8*len(X_train)//batch_size
clr = CyclicLR(base_lr=0.0001, max_lr=0.00001, step_size=step, mode='triangular2')

history = model.fit_generator(generate_arrays_from_file(X_train, Y_train, Y_train_class, batch_size),
                            epochs=num_per_epoch,
                            steps_per_epoch=X_train_size/batch_size,
                            callbacks=[checkpoint])

model.save(_MODEL_FINAL_NAME)
model.save_weights(_MODEL_WEIGHTS_FINAL_NAME)

print("Computing predictions...")

predictions = model.predict_generator(generate_arrays_from_file(X_test, Y_test, Y_train_class, batch_size), steps = len(X_test)//batch_size)

#get_results(predictions, Y_test)

distance_file = "regre-0.5-0.5-NY-0.0001"
results_file = "regre-0.5-0.5-NY-0.0001-results"
print("Computing accuracy...\n")
get_results(distance_file+"-coordinates.results", results_file, predictions, len(Y_test), encoder, res, Y_test, codes_flag=0)
get_results(distance_file+"-codes.results", results_file, predictions, len(Y_test), encoder, res, Y_test, codes_flag=1)
get_results(distance_file+"-mean.results", results_file, predictions, len(Y_test), encoder, res, Y_test, codes_flag=-2)