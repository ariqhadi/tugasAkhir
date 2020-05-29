from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add, MaxPooling2D, Concatenate, Dropout
import math as m

def residual_block(model, kernal_size, filters, strides):

    prev_model = model

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    model = add([prev_model, model])

    return model

def inception_block(model, kernal_size, filters, strides):

    conv_1 = Conv2D(filters = filters, kernel_size = (1,1), strides = strides, padding = "same")(model)

    conv_2 = Conv2D(filters = filters, kernel_size = (1,1), strides = strides, padding = "same")(model)
    conv_2 = Conv2D(filters = filters, kernel_size = (3,3), strides = strides, padding = "same")(conv_2)

    conv_3 = Conv2D(filters = filters, kernel_size = (1,1), strides = strides, padding = "same")(model)
    conv_3 = Conv2D(filters = filters, kernel_size = (5,5), strides = strides, padding = "same")(conv_3)
    
    branches = [conv_1,conv_2,conv_3]
    
    model = Concatenate(axis=-1)(branches)
    return model

def up_sampling_block(model, kernel_size, filters, strides):
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding="same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)
    return model

class Generator(object):
    def __init__(self, LR_shape, factor,arch):
        self.LR_shape = LR_shape
        self.factor = factor
        self.arch = arch

    def generator(self):

        gen_input = Input(shape = self.LR_shape)

        model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

        if self.arch == "resnet":
            gen_model = model
            for index in range(16):
                model = residual_block(model, 3, 64, 1)
            model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
            model = BatchNormalization(momentum = 0.5)(model)
            model = add([gen_model, model])

        if self.arch =="inception":
            for index in range(4):
                model = inception_block(model, 3, 128, 1)
            model = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = "same")(model)

        if self.arch == "inception-resnet":
            gen_model = model
            for index in range(16):
                prev_model = model
                model = inception_block(model, 3, 64, 1)
                model = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = "same")(model)
                model = add([prev_model, model])
            model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
            model = BatchNormalization(momentum = 0.5)(model)
            model = add([gen_model, model])

        upsampling_scale = int(m.log2(self.factor))

        for index in range(upsampling_scale):
            model = up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs = gen_input, outputs = model)
        generator_model.summary()

        return generator_model


class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):

        discriminator_input = Input(shape = self.image_shape)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(discriminator_input)
        model = LeakyReLU(alpha = 0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=discriminator_input, outputs=model)

        return discriminator_model

