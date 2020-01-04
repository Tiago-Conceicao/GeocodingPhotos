from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import latlon2healpix, healpix2latlon, geodistance, mean_confidence_interval, median_confidence_interval
import tensorflow as tf
import numpy as np
import random as rn
import os
import math


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

_PATH = '/home/tconceicao/old_sf_300/'

_TRAIN_TEST_SPLIT_SEED = 2

resolution = math.pow(2, 12)

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

def get_results(mfc, test_instances_sz, resolution, Y1_test):
    distances = []
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    acc_161 = 0
    acc_5 = 0
    acc_1 = 0
    acc_500m = 0
    acc_100m = 0

    coordinates = healpix2latlon(mfc, resolution)
    lat = coordinates[0]
    lon = coordinates[1]

    print(lon,'  ' ,lat)

    for pos in range(test_instances_sz):

      # calculate distance between predicted coordinates and the real ones
      dist = geodistance((Y1_test[pos][0], Y1_test[pos][1]), (lat, lon))
      # to measuse accuracy below threshold
      if dist <= 161:
        acc_161 += 1
      if dist <= 5:
        acc_5 += 1
      if dist <= 1:
        acc_1 += 1
      if dist <= 0.5:
        acc_500m += 1
      if dist <= 0.1:
        acc_100m += 1

      distances.append(dist)
      # for measusing accuracy of region classification using different resolutions
      # comparison of region codes between the real coordinates of the place and the predicted coordinates
      if latlon2healpix(Y1_test[pos][0], Y1_test[pos][1], 4) == latlon2healpix(lat, lon, 4): correct1 += 1
      if latlon2healpix(Y1_test[pos][0], Y1_test[pos][1], math.pow(4, 3)) == latlon2healpix(lat, lon, math.pow(4,
                                                                                                               3)): correct2 += 1
      if latlon2healpix(Y1_test[pos][0], Y1_test[pos][1], math.pow(4, 4)) == latlon2healpix(lat, lon, math.pow(4,
                                                                                                               4)): correct3 += 1
      if latlon2healpix(Y1_test[pos][0], Y1_test[pos][1], math.pow(4, 5)) == latlon2healpix(lat, lon, math.pow(4,
                                                                                                               5)): correct4 += 1

    # print the results and write on a report txt file
    print("Mean distance : %s km" % np.mean(distances))
    print("Median distance : %s km" % np.median(distances))
    print("Confidence interval for mean distance : ", mean_confidence_interval(distances))
    print("Confidence interval for median distance : ", median_confidence_interval(distances))
    print("Region accuracy calculated from coordinates (resolution=4): %s" % float(correct1 / float(len(distances))))
    print("Region accuracy calculated from coordinates (resolution=64): %s" % float(correct2 / float(len(distances))))
    print("Region accuracy calculated from coordinates (resolution=256): %s" % float(correct3 / float(len(distances))))
    print("Region accuracy calculated from coordinates (resolution=1024): %s" % float(correct4 / float(len(distances))))
    print("Accurary under or equal to 161 km:", acc_161 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 5 km:", acc_5 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 1 km:", acc_1 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 500 m:", acc_500m * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 100 m:", acc_100m * 100/ test_instances_sz, "%")
    print()

#Load training images
images, coordinates = get_images(_PATH)


print('IMAGES LOADED')

print("instances:", len(coordinates))


encoder = LabelEncoder()
region_codes = []
for c in (coordinates):
    region = latlon2healpix(float(c[0]), float(c[1]), resolution)
    region_codes.append(region)
classes = to_categorical(encoder.fit_transform(region_codes))

mfc = max(set(region_codes), key=region_codes.count)

coordinates = np.array(coordinates)


print("shape:", classes.shape)

print("Computing accuracy...\n")
get_results(mfc, len(coordinates), resolution, coordinates)