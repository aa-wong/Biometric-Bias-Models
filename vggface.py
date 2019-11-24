
# coding: utf-8

# In[1]:


# Import Dependencies
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from scipy import ndimage


# In[2]:


class VGGFace:
    @staticmethod
    def build(img_dims, weights_path=None):
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

        
        img = Input(shape=input_shape)

        pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
        conv1_1 = Conv2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
        pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
        conv1_2 = Conv2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

        pad2_1 = ZeroPadding2D((1, 1))(pool1)
        conv2_1 = Conv2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
        pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
        conv2_2 = Conv2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

        pad3_1 = ZeroPadding2D((1, 1))(pool2)
        conv3_1 = Conv2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
        pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
        conv3_2 = Conv2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
        pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
        conv3_3 = Conv2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

        pad4_1 = ZeroPadding2D((1, 1))(pool3)
        conv4_1 = Conv2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
        pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
        conv4_2 = Conv2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
        pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
        conv4_3 = Conv2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

        pad5_1 = ZeroPadding2D((1, 1))(pool4)
        conv5_1 = Conv2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
        pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
        conv5_2 = Conv2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
        pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
        conv5_3 = Conv2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

        flat = Flatten()(pool5)
        fc6 = Dense(4096, activation='relu', name='fc6')(flat)
        fc6_drop = Dropout(0.5)(fc6)
        fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
        fc7_drop = Dropout(0.5)(fc7)
        out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

        model = Model(input=img, output=out)

        if weights_path:
            model.load_weights(weights_path)

        return model

