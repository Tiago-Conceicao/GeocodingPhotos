from keras.applications import resnet50
from keras import models, layers, optimizers
from keras.applications import densenet


def classification_model(number_classes):
    #Load the MobileNet model
    image_size=224
    denseNet = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    inp = layers.Input(shape=(image_size, image_size, 3))
    x = denseNet (inp)
    z = layers.Dense(number_classes, activation = 'softmax')(x)
    model = models.Model(inp, z)

    return model


def matrix_regression_model(Y2_train):
    #Load the MobileNet model
    image_size=224
    denseNet = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    input_matrix = layers.Input(shape=(Y2_train.shape[1], 3), dtype='float32', name="matrix")

    inp = layers.Input(shape=(image_size, image_size, 3))
    x = denseNet (inp)
    auxiliary_output = layers.Dense(Y2_train.shape[1], activation='softmax', name = "auxiliary_output")(x)

    main_output = layers.dot([auxiliary_output, input_matrix], axes=1, name='main_output')

    model = models.Model([inp, input_matrix], [main_output, auxiliary_output])

    return model


def regre_class_model(number_classes):
    #Load the MobileNet model
    image_size=224
    denseNet = densenet.DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3), pooling = 'avg')

    #build model
    inp = layers.Input(shape=(224, 224, 3))
    x = denseNet (inp)
    z = layers.Dense(2, activation = 'sigmoid')(x)
    q = layers.Dense(number_classes, activation = 'softmax')(x)
    model = models.Model(inp, [z,q])

    return model
