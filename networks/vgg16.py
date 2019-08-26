# -*- coding: utf-8 -*-
"""
VGG network
"""

from keras.models import Model
from keras.layers import Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras import backend as K
from keras.applications.vgg16 import VGG16

seed_number = 1
# Non-linearity params
leakiness = 0.0
# Batchnorm params
mom = 0.99
eps = 0.001
gamma = 'ones'
# Convolution params
bias = True
weight_decay = 0.0005
initer = initializers.he_normal(seed=seed_number)

def create_model(input_shape):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_tensor = Input(shape=input_shape)
    base_model = VGG16()
    x = base_model.output
    x = BatchNormalization(axis=channel_axis, momentum=mom,
                           epsilon=eps, gamma_initializer=gamma)(x)
    x = LeakyReLU(leakiness)(x)
    model = Model(input_tensor,x,name = 'vgg16_trunk')
    return model