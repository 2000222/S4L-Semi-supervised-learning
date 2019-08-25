# -*- coding: utf-8 -*-
"""ResNet50 model """
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
from keras.layers import GlobalAveragePooling2D
from keras import applications
from keras.applications.resnet50 import ResNet50

seed_number = 1
# Non-linearity params
leakiness = 0.0
# Batchnorm params
mom = 0.99
eps = 0.001
gamma = 'ones'
# Convolution params
bias = False
weight_decay = 0.0005
initer = initializers.he_normal(seed=seed_number)

def create_model(input_shape):
    #load the model
	#module=hub.module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3")
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(include_top=False,
                   weights='imagenet',
                   input_tensor=input_tensor)
    #base_model.load_weights('/content/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = base_model.output
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    model = Model(input_tensor,x,name = 'resnet50_trunk')
    return model
