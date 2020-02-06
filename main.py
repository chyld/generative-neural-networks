from functools import reduce
from operator import mul
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def eda():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test, y_train, y_test

def dense_encoder(shape, neurons, dim):
    i = Input(shape=shape)
    x = Flatten()(i)
    x = Dense(units=neurons)(x)
    x = LeakyReLU()(x)
    x = Dense(units=dim)(x)
    x = LeakyReLU()(x)
    return Model(i, x, name='Dense-Encocder')

def conv_encoder(shape, neurons, dim):
    i = Input(shape=shape)
    x = Conv2D(32, padding='same', kernel_size=(3,3), activation='relu')(i)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64, padding='same', kernel_size=(3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(units=neurons)(x)
    x = LeakyReLU()(x)
    x = Dense(units=dim)(x)
    x = LeakyReLU()(x)
    return Model(i, x, name='Conv-Encoder')

def dense_decoder(shape, neurons, dim):
    flattened = reduce(mul, shape)
    i = Input(shape=dim)
    x = Dense(units=neurons)(i)
    x = LeakyReLU()(x)
    x = Dense(units=flattened)(x)
    x = LeakyReLU()(x)
    x = Reshape(shape)(x)
    return Model(i, x, name='Dense-Decoder')

def conv_decoder(shape, neurons, dim):
    # flattened = reduce(mul, shape)
    i = Input(shape=dim)
    x = Dense(units=neurons)(i)
    x = LeakyReLU()(x)
    x = Dense(units=3136)(x)
    x = LeakyReLU()(x)
    x = Reshape((7, 7, 64))(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, padding='same', kernel_size=(3,3), activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, padding='same', kernel_size=(3,3), activation='relu')(x)
    x = Conv2D(1, padding='same', kernel_size=(3,3), activation='relu')(x)
    return Model(i, x, name='Conv-Decoder')
