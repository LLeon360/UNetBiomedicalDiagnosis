# data vis and processing
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

# for building model 
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

class Math:
  def add(a, b):
    return a + b
  def subtract(a,b):
    return a - b
  
class Unet:
  @classmethod
  def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 
                  3,     
                  activation='relu',
                  padding='same')(inputs)
    conv = Conv2D(n_filters, 
                  3,   
                  activation='relu',
                  padding='same')(conv)

    conv = BatchNormalization()(conv, training=False)

    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv

    return next_layer, skip_connection

  @classmethod
  def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(
                  n_filters,
                  (3,3),   
                  strides=(2,2),
                  padding='same')(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(n_filters, 
                  3,     
                  activation='relu',
                  padding='same')(merge)
    conv = Conv2D(n_filters,
                  3,  
                  activation='relu', 
                  padding='same')(conv)
    return conv

  @classmethod
  def BuildModel(input_size=(128, 128, 1), n_filters=32, n_classes=2, n_layers = 4, dropout=0.3):

    inputs = Input(input_size)

    eblocks = []
    #n_layers is encoder layers
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
    eblocks.append(Unet.EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True))
    for i in range(n_layers-1):
      dropout = 0
      if(i > 1):
        dropout = 0.3
      eblocks.append(Unet.EncoderMiniBlock(eblocks[-1][0], n_filters*(2**(i+1)), dropout_prob=dropout, max_pooling=True))

    #intermediate latent also put in encoder
    eblocks.append(Unet.EncoderMiniBlock(eblocks[-1][0], n_filters*(2**(n_layers)), dropout_prob=0, max_pooling=False)) 

    dblocks = []
    dblocks.append(Unet.DecoderMiniBlock(eblocks[-1][0], eblocks[-2][1],  n_filters*(2**(n_layers-1))))
    for i in range(n_layers-1): 
      dblocks.append(Unet.DecoderMiniBlock(dblocks[-1], eblocks[-i-3][1],  n_filters*(2**(n_layers-2-i))))

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv_last = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same')(dblocks[-1])

    out = Conv2D(n_classes, 1, 
                  activation='softmax', padding='same')(conv_last)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model

#   def UNet(input_size=(128, 128, 1), 
#            n_filters=32, 
#            n_classes=2, 
#            n_layers = 4, 
#            dropout=0.3,
#            input,

#           ):
  

# class ImageProcessing:
  
