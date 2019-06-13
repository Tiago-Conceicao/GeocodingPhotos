import numpy as np
from keras import backend as K
import tensorflow as tf
from geopy import distance
import math
import healpy
from scipy import stats

_BOUNDING_BOX = [37.639830, 37.929824, -123.173825, -122.281780] #sf
#_BOUNDING_BOX = [40.477399,  40.917577, -74.259090, -73.700272] #ny

def geodistance_theano( p1 , p2 ):  #p1: lat, lon; p2: lat, lon
  a0 = convertvalues(p1[:,0], _BOUNDING_BOX[0], _BOUNDING_BOX[1])
  a1 = convertvalues(p1[:,1], _BOUNDING_BOX[2],  _BOUNDING_BOX[3])
  b0 = convertvalues(p2[:,0], _BOUNDING_BOX[0] , _BOUNDING_BOX[1])
  b1 = convertvalues(p2[:,1], _BOUNDING_BOX[2],  _BOUNDING_BOX[3])

  aa0 = a0 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  aa1 = a1 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  bb0 = b0 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0
  bb1 = b1 * 3.141592653589793238462643383279502884197169399375105820974944592307816406286 / 180.0

  sin_lat1 = K.sin( aa0 )
  cos_lat1 = K.cos( aa0 )
  sin_lat2 = K.sin( bb0 )
  cos_lat2 = K.cos( bb0 )
  delta_lng = bb1 - aa1
  cos_delta_lng = K.cos(delta_lng)
  sin_delta_lng = K.sin(delta_lng)
  d = tf.atan2(K.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
  return K.mean( 6371.0087714 * d , axis = -1 )

def geodistance_tensorflow( p1 , p2 ): #p1: lat, lon; p2: x, y, z
    aa0 = p1[:,0] * 0.01745329251994329576924
    aa1 = p1[:,1] * 0.01745329251994329576924
    bb0 = tf.atan2(p2[:,2], K.sqrt(p2[:,0] ** 2 + p2[:,1] ** 2)) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
    bb1 = tf.atan2(p2[:,1], p2[:,0]) * 180.0 / 3.141592653589793238462643383279502884197169399375105820974944592307816406286
    bb0 = bb0 * 0.01745329251994329576924
    bb1 = bb1 * 0.01745329251994329576924
    sin_lat1 = K.sin( aa0 )
    cos_lat1 = K.cos( aa0 )
    sin_lat2 = K.sin( bb0 )
    cos_lat2 = K.cos( bb0 )
    delta_lng = bb1 - aa1
    cos_delta_lng = K.cos(delta_lng)
    sin_delta_lng = K.sin(delta_lng)
    d = tf.atan2(K.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
    return K.mean( 6371.0087714 * d , axis = -1 )

def normalize_values (x, minv, maxv):
    return float(((float(x) - float(minv))/(float(maxv)-float(minv))))

def convertvalues(x, minv, maxv):
    return (x)*(maxv-minv)+minv

def geodistance(coords1, coords2):
  lat1, lon1 = coords1[: 2]
  lat2, lon2 = coords2[: 2]

  try:
    return distance.vincenty((lat1, lon1), (lat2, lon2)).meters / 1000.0
  except:
    return distance.great_circle((lat1, lon1), (lat2, lon2)).meters / 1000.0

# confidence intervals for the mean error values
def mean_confidence_interval( data ):
    confidence = 0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a) , stats.sem(a)
    h = se * stats.t._ppf( ( 1 + confidence ) / 2.0 , n-1 )
    return m , m-h , m+h , h

# confidence intervals for the median error values
def median_confidence_interval( data ):
    n_boots = 10000
    sample_size = 50
    a = 1.0 * np.array(data)
    me = [ ]
    np.random.seed(seed=0)
    for _ in range(0,n_boots):
        sample = [ a[ np.random.randint( 0 , len(data) - 1 ) ] for _ in range(0,sample_size) ]
        me.append( np.median( sample ) )
    med = np.median(data)
    mph = np.percentile(me, 2.5)
    mmh = np.percentile(me, 97.5)
    return med , mph , mmh

def latlon2healpix(lat, lon, res):
  lat = float(lat)
  lon = float(lon)
  lat = lat * math.pi / 180.0
  lon = lon * math.pi / 180.0
  xs = (math.cos(lat) * math.cos(lon))
  ys = (math.cos(lat) * math.sin(lon))
  zs = (math.sin(lat))
  return healpy.vec2pix(int(res), xs, ys, zs)

def healpix2latlon(code, res):
  [xs, ys, zs] = healpy.pix2vec(int(res), code)
  lat = float(math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi)
  lon = float(math.atan2(ys, xs) * 180.0 / math.pi)
  return [lat, lon]

def get_results(distance_file_name, results_file, predictions, test_instances_sz, encoder, resolution, Y1_test, codes_flag):
    distances = []
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    f = open(distance_file_name, 'w')

    # get the correspondent predictions of the region class
    # since it is a probability vector (size of number of classes)
    # region class is the one with higher probability
    y_classes = predictions[1].argmax(axis=-1)
    # since we encoded the class as categorical classes
    # we nedd to inverse_tranform to get the real code class
    y_classes = encoder.inverse_transform(y_classes)

    acc_5 = 0
    acc_1 = 0
    acc_500m = 0
    acc_100m = 0
    for pos in range(test_instances_sz):
      if codes_flag == 1:
        # codes_flag = 1: results based on centroide coordinates of the region predicted
        # convert the healpix code back to latitude and longitude coordinates, given the resolution
        coordinates = healpix2latlon(y_classes[pos], resolution)
        lat = coordinates[0]
        lon = coordinates[1]

      elif codes_flag == 0:
        # codes_flag = 0: results based on coordinates predictions only
        (xs, ys, zs) = predictions[0][pos]
        # conversion of the cartesian geographic coordinates (x,y,z) into the latitude and longitude coordinates

        lat = float(math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi)
        lon = float(math.atan2(ys, xs) * 180.0 / math.pi)

      elif codes_flag == -2:
        # codes_flag = -2: results based on mean of both coordinates prediction and centroide of region prediction
        # equal to the 2 previous cases, but we use the mean of both to get the results
        (xs1, ys1, zs1) = predictions[0][pos]

        coordinates = healpix2latlon(y_classes[pos], resolution)
        lat_aux = coordinates[0] * math.pi / 180.0
        lon_aux = coordinates[1] * math.pi / 180.0
        xs2 = (math.cos(lat_aux) * math.cos(lon_aux))
        ys2 = (math.cos(lat_aux) * math.sin(lon_aux))
        zs2 = (math.sin(lat_aux))

        xs = (xs1 + xs2) / 2
        ys = (ys1 + ys2) / 2
        zs = (zs1 + zs2) / 2
        lat = float(math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi)
        lon = float(math.atan2(ys, xs) * 180.0 / math.pi)

      # calculate distance between predicted coordinates and the real ones
      dist = geodistance((Y1_test[pos][0], Y1_test[pos][1]), (lat, lon))
      # to measuse accuracy bellow 161km
      if dist <= 5:
        acc_5 += 1
      if dist <= 1:
        acc_1 += 1
      if dist <= 0.5:
        acc_500m += 1
      if dist <= 0.1:
        acc_100m += 1
      # regist the predicted results for test split
      #        f.write(test_names[pos].replace(" ","_") + "\t" + str(lat) + "\t" + str(lon) + "\t" + str(Y1_test[pos][0]) + "\t" + str(Y1_test[pos][1]) + "\t" + str(dist) + "\n")
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
    print("Accurary under or equal to 5 km:", acc_5 / test_instances_sz)
    print("Accurary under or equal to 1 km:", acc_1 / test_instances_sz)
    print("Accurary under or equal to 500 m:", acc_500m / test_instances_sz)
    print("Accurary under or equal to 100 m:", acc_100m / test_instances_sz)
    print()
    f.close()

    '''
    out_results = open(results_file, "a")
    out_results.write("Mean distance : %s km\n" % np.mean(distances))
    out_results.write("Median distance : %s km\n" % np.median(distances))
    out_results.write("Confidence interval for mean distance : " + str(mean_confidence_interval(distances)) + "\n")
    out_results.write("Confidence interval for median distance : " + str(median_confidence_interval(distances)) + "\n")
    out_results.write(
      "Region accuracy calculated from coordinates (resolution=4): %s\n" % float(correct1 / float(len(distances))))
    out_results.write(
      "Region accuracy calculated from coordinates (resolution=64): %s\n" % float(correct2 / float(len(distances))))
    out_results.write(
      "Region accuracy calculated from coordinates (resolution=256): %s\n" % float(correct3 / float(len(distances))))
    out_results.write(
      "Region accuracy calculated from coordinates (resolution=1024): %s\n" % float(correct4 / float(len(distances))))
    out_results.write("Accurary under or equal to 161km: " + str(acc_161 / test_instances_sz) + "\n\n")
    out_results.close()
    '''

def square(vector):
    new_vector = (vector**2.0)/(K.sum(vector**2.0))
    return new_vector

def cube(vector):
    new_vector = (vector**3.0)/(K.sum(vector**3.0))
    return new_vector