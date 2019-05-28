import os
import math
import pickle
import healpy
import keras
import time
import string
import random
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from ast import literal_eval
from elmoformanylangs import Embedder
from clr_callback import CyclicLR
from geopy import distance
from scipy import stats
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.layers.core import Reshape
from keras.layers import concatenate, dot
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

# loss function corresponding to penalized hyperbolic tangent
class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'

    def call(self, inputs): return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))

    def get_config(self): return super(Pentanh, self).get_config()

    def compute_output_shape(self, input_shape): return input_shape

keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})

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

# convert latidude and longitude to a healpix code encoding a region, with a given resolution 
# the resultion is defined in main   
def latlon2healpix( lat , lon , res ):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    xs = ( math.cos(lat) * math.cos(lon) )
    ys = ( math.cos(lat) * math.sin(lon) )
    zs = ( math.sin(lat) )
    return healpy.vec2pix( int(res) , xs , ys , zs )

# convert healpix code of a given resolution, back into latitude and longitude coordinates
# the resultion is defined in main   
def healpix2latlon( code , res ):
    [xs, ys, zs] = healpy.pix2vec( int(res) , code )
    lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
    lon = float( math.atan2(ys, xs) * 180.0 / math.pi )
    return [ lat , lon ]

# return geodesic distance between two points    
def geodistance( coords1 , coords2 ):
    lat1 , lon1 = coords1[ : 2]
    lat2 , lon2 = coords2[ : 2]

    try: return distance.vincenty( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
    except: return distance.great_circle( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0

# tensorflow function for computing the great circle distance (geodesic distance) between two points
def geodistance_tensorflow( p1 , p2 ):
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

######
# text pre-processing - before calculating the embeddings

def normalize_size(words, size):
    if len(words) == size:
        return words
    if len(words) > size:
        return words[:size]
    else:
        missing = size-len(words)
        words += [" "]*missing
        return words

def find_sentence_index(sentence, text, current):
    index = text.index(sentence[0], current)
    for i in range(len(sentence)):
        if sentence[i] != text[index+i]:
            return find_sentence_index(sentence, text, index+1)
    return index

def get_text_words(mention, sentence, text, size):
    text_size = len(text)
    if text_size<=size:
        missing = size-text_size
        text += [" "]*missing
        return text
    else:
        start_sent_idx = find_sentence_index(sentence, text, 0)
        start_mention_idx = text.index(mention[0], start_sent_idx)

        left_side = round(size/2)
        if (start_mention_idx-left_side)>=0:
            words = text[start_mention_idx-left_side:start_mention_idx+len(mention)]
        else:
            words = text[0:start_mention_idx+len(mention)]

        rigth_side = size-len(words)
        if (start_mention_idx+len(mention)+rigth_side) <= text_size:
            words += text[start_mention_idx+len(mention):start_mention_idx+len(mention)+rigth_side]
        else:
            words += text[start_mention_idx+len(mention):text_size]
            if len(words)<size:
                to_add_begin = size-len(words)
                words = text[text_size-len(words)-to_add_begin:text_size-len(words)] + words
        return words

######

def elmo_embeddings(data_file, file_name):

    if not Path(file_name).is_file():
        i=0
        pickle_file = open(file_name, "ab")
        num_lines = sum(1 for line in open(data_file))

        with open(data_file, "r") as file:
            coord_data = []
            mention_data = []
            sentence_data = []
            text_data = []
            translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            for line in file:
                if i%10==0:
                    print("ITERATION: "+str(i))
                i+=1

                content = line.split("\t")
                mention = content[0].lower().translate(translator)
                mention = re.sub("[–’“”]"," ", mention)
                mention = mention.rstrip("\n").split(" ")
                mention = list(filter(None, mention))

                sentence = content[2].lower().translate(translator)
                sentence = re.sub("[–’“”]"," ", sentence)
                sentence = sentence.rstrip("\n").split(" ")
                sentence = list(filter(None, sentence))

                text = content[3].lower().translate(translator)
                text = re.sub("[–’“”]"," ", text)
                text = text.rstrip("\n").split(" ")
                text = list(filter(None, text))
                text_vector = get_text_words(mention, sentence, text, text_size)

                mention_vector = normalize_size(mention, mention_size) 
                sentence_vector = normalize_size(sentence, sentence_size)
                
                #Mention
                mention_embed = elmo.sents2elmo([mention_vector])
                mention_embed = np.array(mention_embed[0])
                #Sentence
                sentence_embed = elmo.sents2elmo([sentence_vector])
                sentence_embed = np.array(sentence_embed[0])
                #Text
                text_embed = elmo.sents2elmo([text_vector])
                text_embed = np.array(text_embed[0])
                
                coordinates = literal_eval(content[1])
                coord_data.append(coordinates)
                
                mention_data.append(mention_embed)
                sentence_data.append(sentence_embed)
                text_data.append(text_embed)

                # write in sizes of batch size to pickle file 
                if i%batch_size==0:
                    pickle.dump([mention_data, sentence_data, text_data], pickle_file)
                    mention_data = []
                    sentence_data = []
                    text_data = []
            
                if i==num_lines:
                    pickle.dump([mention_data, sentence_data, text_data], pickle_file)
                
            pickle_file.close()
            file.close()
            return coord_data
    else:
        coord_data = []
        with open(data_file, "r") as file:
            for line in file:
                content = line.split("\t")
                coord_data.append(literal_eval(content[1]))
        return coord_data

def batch_generator(file_name, size, matrix, Y1, Y2):
    while 1:
        file=open(file_name, 'rb')
        for i in range(0,size,batch_size):
            aux = pickle.load(file)
            X1 = np.asarray(aux[0])
            X2 = np.asarray(aux[1])
            X3 = np.asarray(aux[2])
            region_matrix = matrix*len(X1)
            region_matrix = np.asarray(region_matrix)
            yield [X1, X2, X3, region_matrix],[Y1[i:i+batch_size], Y2[i:i+batch_size]]
        file.close()

def get_mentions(file_name):
    mentions = []

    with open(file_name, "r") as file:
        for line in file:
            content = line.split("\t")
            mentions.append(content[0])
        file.close()
    return mentions

def get_results(distance_file_name, results_file, predictions, test_instances_sz, codes_flag):
    distances = [ ]
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

    acc_161 = 0
    for pos in range(test_instances_sz):
        if codes_flag == 1:
            # codes_flag = 1: results based on centroide coordinates of the region predicted
            # convert the healpix code back to latitude and longitude coordinates, given the resolution
            coordinates = healpix2latlon(y_classes[pos], resolution)
            lat = coordinates[0]
            lon = coordinates[1]

        elif codes_flag == 0:
            # codes_flag = 0: results based on coordinates predictions only
            ( xs, ys, zs ) = predictions[0][pos]
            # conversion of the cartesian geographic coordinates (x,y,z) into the latitude and longitude coordinates
            lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
            lon = float( math.atan2(ys, xs) * 180.0 / math.pi )

        elif codes_flag == -2:
            # codes_flag = -2: results based on mean of both coordinates prediction and centroide of region prediction
            # equal to the 2 previous cases, but we use the mean of both to get the results
            ( xs1, ys1, zs1 ) = predictions[0][pos]

            coordinates = healpix2latlon(y_classes[pos], resolution)
            lat_aux = coordinates[0] * math.pi / 180.0
            lon_aux = coordinates[1] * math.pi / 180.0
            xs2 = ( math.cos(lat_aux) * math.cos(lon_aux) )
            ys2 = ( math.cos(lat_aux) * math.sin(lon_aux) )
            zs2 = ( math.sin(lat_aux) )

            xs = (xs1 + xs2) / 2
            ys = (ys1 + ys2) / 2
            zs = (zs1 + zs2) / 2
            lat = float( math.atan2(zs, math.sqrt(xs * xs + ys * ys)) * 180.0 / math.pi )
            lon = float( math.atan2(ys, xs) * 180.0 / math.pi )

        # calculate distance between predicted coordinates and the real ones
        dist = geodistance( ( Y1_test[pos][0] , Y1_test[pos][1] ) , ( lat , lon ) )
        # to measuse accuracy bellow 161km 
        if dist <=161:
            acc_161 += 1
        # regist the predicted results for test split
#        f.write(test_names[pos].replace(" ","_") + "\t" + str(lat) + "\t" + str(lon) + "\t" + str(Y1_test[pos][0]) + "\t" + str(Y1_test[pos][1]) + "\t" + str(dist) + "\n")
        distances.append( dist )
        # for measusing accuracy of region classification using different resolutions
        # comparison of region codes between the real coordinates of the place and the predicted coordinates
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , 4 ) == latlon2healpix ( lat, lon , 4 ): correct1 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 3) ) == latlon2healpix ( lat, lon , math.pow(4 , 3) ): correct2 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 4) ) == latlon2healpix ( lat, lon , math.pow(4 , 4) ): correct3 += 1
        if latlon2healpix ( Y1_test[pos][0] , Y1_test[pos][1] , math.pow(4 , 5) ) == latlon2healpix ( lat, lon , math.pow(4 , 5) ): correct4 += 1

    # print the results and write on a report txt file
    print("Mean distance : %s km" % np.mean(distances) )
    print("Median distance : %s km" % np.median(distances) )
    print("Confidence interval for mean distance : " , mean_confidence_interval(distances) )
    print("Confidence interval for median distance : " , median_confidence_interval(distances) )
    print("Region accuracy calculated from coordinates (resolution=4): %s" % float(correct1 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=64): %s" % float(correct2 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=256): %s" % float(correct3 / float(len(distances))) )
    print("Region accuracy calculated from coordinates (resolution=1024): %s" % float(correct4 / float(len(distances))) )
    print("Accurary under or equal to 161km:", acc_161/test_instances_sz)
    print()
    f.close()

    out_results = open(results_file,"a")
    out_results.write("Mean distance : %s km\n" % np.mean(distances))
    out_results.write("Median distance : %s km\n" % np.median(distances))
    out_results.write("Confidence interval for mean distance : " + str(mean_confidence_interval(distances))+"\n")
    out_results.write("Confidence interval for median distance : " + str(median_confidence_interval(distances))+"\n")
    out_results.write("Region accuracy calculated from coordinates (resolution=4): %s\n" % float(correct1 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=64): %s\n" % float(correct2 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=256): %s\n" % float(correct3 / float(len(distances))))
    out_results.write("Region accuracy calculated from coordinates (resolution=1024): %s\n" % float(correct4 / float(len(distances))))
    out_results.write("Accurary under or equal to 161km: "+str(acc_161/test_instances_sz)+"\n\n")
    out_results.close()

if __name__ == '__main__':

    start = time.time()

    train_data = "../wotr-corpus/wotr-topo-train.txt"
    test_data = "../wotr-corpus/wotr-topo-test.txt"
    pickle_train = "../wotr-corpus/train-500words.pickle"
    pickle_test = "../wotr-corpus/test-500words.pickle"
    model_name = '70-30-wotr-geo-model.h5'
    distance_file = '70-30-wotr-geo-model'
    results_file = "70-30-wotr-results.txt"

    mention_size = 5
    sentence_size = 50
    text_size = 500
    embedding_size = 1024
    # resolution used in healpix
    # the higher the resolution, the more regions on the earth's surface (each of them with a smaller area)
    resolution = math.pow(4,4)
    batch_size = 32
    epochs = 200
    
    elmo = Embedder("144/", batch_size=batch_size)

    # embedding calculation, saves the results into a pickle file
    # only if the file does not exist, if exists uses them (calculated previously)
    print("Train dataset - ELMO embedding: Sarting computation...")
    Y1_train = elmo_embeddings(train_data, pickle_train)
    print("Train dataset - ELMO embedding: Computation finished.")

    print("Test dataset - ELMO embedding: Sarting computation...")
    Y1_test = elmo_embeddings(test_data, pickle_test)
    print("Test dataset - ELMO embedding: Computation finished.")

    train_instances_sz = len(Y1_train)
    test_instances_sz = len(Y1_test)
    print("Train instances:", train_instances_sz)
    print("Test instances:", test_instances_sz)

    test_names = get_mentions(test_data)

    encoder = LabelEncoder()
    region_codes = []
    # for the dataset, get the coordinates (latitude and longitude)
    # and convert each instance's coordinates into the correspondent healpix code
    # given the resolution defined above
    for coordinates in (Y1_train+Y1_test):
        region = latlon2healpix(coordinates[0], coordinates[1], resolution)
        region_codes.append(region)
    # Encoded the region codes into categorical classes
    classes = to_categorical(encoder.fit_transform(region_codes))

    Y1_train = np.array(Y1_train)
    Y1_test = np.array(Y1_test)

    Y2_train = classes[:train_instances_sz]
    Y2_test = classes[-test_instances_sz:]

    print("Build model...")

    print("Y1_train shape:",Y1_train.shape)
    print("Y1_test shape:",Y1_test.shape)
    print("Y2_train shape:",Y2_train.shape)
    print("Y2_test shape:",Y2_test.shape)
    
    # constuction of a matrix, where each row is the centroide coordinates (x,y,z)
    # that corresponds to each diferent region class
    # number of rows is equal to the number of existing region classes
    region_list = [i for i in range(Y2_train.shape[1])]
    region_classes = encoder.inverse_transform(region_list)
    codes_matrix = []
    for i in range(len(region_classes)):
        [xs, ys, zs] = healpy.pix2vec( int(resolution), region_classes[i] )
        codes_matrix.append([xs, ys, zs])
    codes_matrix=[codes_matrix]

    # model architecture
    input_mention = Input(shape=(mention_size, embedding_size), dtype='float32', name="mention")
    input_sentence = Input(shape=(sentence_size, embedding_size), dtype='float32', name="sentence")
    input_text = Input(shape=(text_size, embedding_size), dtype='float32', name="text")
    input_matrix = Input(shape=(Y2_train.shape[1], 3), dtype='float32', name="matrix")

    x1 = Bidirectional(LSTM(512, activation='pentanh', recurrent_activation='pentanh'))(input_mention)
    x2 = Bidirectional(LSTM(512, activation='pentanh', recurrent_activation='pentanh'))(input_sentence)
    x3 = Bidirectional(LSTM(512, activation='pentanh', recurrent_activation='pentanh'))(input_text)

    concat1 = concatenate([x1, x2, x3], axis=1)
    x = Dense(512, activation='pentanh')(concat1)

    # outputs:
    # auxiliary_output = region codes probabilities
    auxiliary_output = Dense(Y2_train.shape[1], activation='softmax', name='aux_output')(x)
    # interpolation of auxiliary_output and the matriz of coordinates of the centroides of the regions
    main_output = dot([auxiliary_output, input_matrix], axes=1, name='main_output')

    model = Model(inputs=[input_mention, input_sentence, input_text, input_matrix], outputs=[main_output, auxiliary_output])

    model.summary()
    # training
    print("Training the model...")
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    clr = CyclicLR(base_lr=0.00001, max_lr=0.0001, mode='triangular', step_size=(2.0 - 8.0) * (len(Y1_train)/epochs))
    adamOpt = keras.optimizers.Adam(lr=0.0001, amsgrad=True, clipnorm=1.)

    model.compile(optimizer=adamOpt, loss={"main_output": geodistance_tensorflow, "aux_output": 'categorical_crossentropy'}, loss_weights=[0.70, 30.0], metrics={"main_output": geodistance_tensorflow, "aux_output": 'categorical_accuracy'})
    model.fit_generator( batch_generator(pickle_train, train_instances_sz, codes_matrix, Y1_train, Y2_train),
            steps_per_epoch=train_instances_sz/batch_size, epochs=epochs, callbacks=[clr, early_stop],
            validation_steps=test_instances_sz/batch_size, validation_data=batch_generator(pickle_test, test_instances_sz, codes_matrix, Y1_test, Y2_test))

    model.save(model_name)
    # load a training model
    # model = load_model(model_name, custom_objects={'geodistance_tensorflow': geodistance_tensorflow})
    print("Computing predictions...")

    predictions = model.predict_generator(batch_generator(pickle_test, test_instances_sz, codes_matrix, Y1_test, Y2_test), steps=test_instances_sz/batch_size)

    print("Computing accuracy...\n")
    # get the results calculated diferent forms:
    # codes_flag = 0: from the coordinates predictions only
    # codes_flag = 1: from the centroide coordinates of the region predicted
    # codes_flag = -2: from the mean of both coordinates prediction and centroide of region prediction
    # see the implemation and comments of this function, to see more of the healpix use
    get_results(distance_file+"-coordinates.results", results_file, predictions, test_instances_sz, codes_flag=0)
    get_results(distance_file+"-codes.results", results_file, predictions, test_instances_sz, codes_flag=1)
    get_results(distance_file+"-mean.results", results_file, predictions, test_instances_sz, codes_flag=-2)

    end = time.time()
    print("Execution time:",round((end-start)/60,5)/60,"hours.")
    out_results = open(results_file,"a")
    out_results.write("Execution time:"+str(round((end-start)/60,5)/60)+"hours.")
    out_results.close()
