
# coding: utf-8

# In[1]:


# Import Dependencies
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from scipy import ndimage


# In[2]:


class SmallerVGGNet:
    @staticmethod
    def build(img_dims, classes, final_act="softmax"):
        # Initialize the model
        model = Sequential()
        input_shape = (img_dims[1],
                       img_dims[0],
                       img_dims[2])
        chan_dim = -1

        # Update input_shape if using channels_first
        if K.image_data_format() == "channels_first":
            input_shape = (img_dims[2],
                           img_dims[1],
                           img_dims[0])
            chan_dim = 1


        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape, name='conv2d_1'))
        model.add(Activation("relu", name="activation_1"))
        model.add(BatchNormalization(axis=chan_dim, name='batch_normalization_1'))
        model.add(MaxPooling2D(pool_size=(3, 3), name='max_pooling_2d_1'))
        model.add(Dropout(0.25, name='dropout_1'))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same", name='conv2d_2'))
        model.add(Activation("relu", name="activation_2"))
        model.add(BatchNormalization(axis=chan_dim, name='batch_normalization_2'))
        model.add(Conv2D(64, (3, 3), padding="same", name='conv2d_3'))
        model.add(Activation("relu", name="activation_3"))
        model.add(BatchNormalization(axis=chan_dim, name='batch_normalization_3'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_2d_2'))
        model.add(Dropout(0.25, name='dropout_2'))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same", name='conv2d_4'))
        model.add(Activation("relu", name="activation_4"))
        model.add(BatchNormalization(axis=chan_dim, name='batch_normalization_4'))
        model.add(Conv2D(128, (3, 3), padding="same", name='conv2d_5'))
        model.add(Activation("relu", name="activation_5"))
        model.add(BatchNormalization(axis=chan_dim, name='batch_normalization_5'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling_2d_3'))
        model.add(Dropout(0.25, name='dropout_3'))

        # first (and only) set of FC => RELU layers
        model.add(Flatten(name='flatten_1'))
        model.add(Dense(1024, name='dense_1'))
        model.add(Activation("relu", name="activation_6"))
        model.add(BatchNormalization(name='batch_normalization_6'))
        model.add(Dropout(0.5, name='dropout_4'))

        # use a *softmax* activation for single-label classification
        # and *sigmoid* activation for multi-label classification
        model.add(Dense(classes, name='dense_2'))
        model.add(Activation(final_act, name="activation_7"))

        # return the constructed network architecture
        return model
