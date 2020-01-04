from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import latlon2healpix, healpix2latlon, geodistance, mean_confidence_interval, median_confidence_interval
import tensorflow as tf
import numpy as np
import random as rn
import os
import math
from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy.stats import kde


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.3
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

#_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
_BOUNDING_BOX = [37.639830, 37.929824, -122.7278025, -122.281780] #sf
#_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny

_PATH = '/home/tconceicao/old_sf_300/'
#_PATH = '/home/tconceicao/flickr_ny_300/'
#_PATH = '/home/tconceicao/col+flickr_sf/'


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

def get_results(predictions, test_instances_sz, Y1_test):
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

    for pos in range(test_instances_sz):

      # calculate distance between predicted coordinates and the real ones
      dist = geodistance((Y1_test[pos][0], Y1_test[pos][1]), (predictions[pos][0], predictions[pos][1]))
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


    # print the results and write on a report txt file
    print("Mean distance : %s km" % np.mean(distances))
    print("Median distance : %s km" % np.median(distances))
    print("Confidence interval for mean distance : ", mean_confidence_interval(distances))
    print("Confidence interval for median distance : ", median_confidence_interval(distances))
    print("Accurary under or equal to 161 km:", acc_161 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 5 km:", acc_5 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 1 km:", acc_1 * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 500 m:", acc_500m * 100/ test_instances_sz, "%")
    print("Accurary under or equal to 100 m:", acc_100m * 100/ test_instances_sz, "%")
    print()

#Load training images
images, coordinates = get_images(_PATH)

lats = []
lons = []
for i in range (len(coordinates)):
    lats.append(float(coordinates[i][0]))
    lons.append(float(coordinates[i][1]))

lats = np.array(lats)
lons = np.array(lons)

print('IMAGES LOADED')

print("instances:", len(coordinates))

fig = plt.figure(figsize=(16, 16))
m = Basemap(llcrnrlat=_BOUNDING_BOX[0], urcrnrlat=_BOUNDING_BOX[1], llcrnrlon=_BOUNDING_BOX[2],
            urcrnrlon=_BOUNDING_BOX[3], resolution='f')


lon, lat = m(lons, lats)
m.drawcoastlines(zorder=10)
m.drawmapboundary(fill_color='w')
m.fillcontinents(color='navajowhite',lake_color='w')


nbins = int(math.sqrt(len(coordinates)))
x = lons
y = lats
k = kde.gaussian_kde([x, y])
xi, yi = np.mgrid[_BOUNDING_BOX[2]:_BOUNDING_BOX[3]:nbins * 1j, _BOUNDING_BOX[0]:_BOUNDING_BOX[1]:nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5, zorder=5)

#m.scatter(lons, lats, marker = ',', color='r', zorder=6)
plt.savefig('/home/tconceicao/mapgt.png')
print("done")
'''
l1 = abs(_BOUNDING_BOX[1] - _BOUNDING_BOX[0])
l2 = abs(_BOUNDING_BOX[3] - _BOUNDING_BOX[2])
lat = 0
lon = 0
predictions = []
for i in images:
    lat = _BOUNDING_BOX[0] + l1 * rn.random()
    lon = _BOUNDING_BOX[2] + l2 * rn.random()
    predictions.append([lat, lon])

coordinates = np.array(coordinates)


print("Computing accuracy...\n")
get_results(predictions, len(coordinates), coordinates)
'''