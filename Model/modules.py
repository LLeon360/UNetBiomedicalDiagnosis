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

from keras import backend as K
  
def EncoderBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
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

def DecoderBlock(prev_layer_input, skip_layer_input, n_filters=32):
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

def build_model(input_size=(128, 128, 1), n_filters=32, n_classes=2, n_layers = 4, dropout=0.3):
  inputs = Input(input_size)

  eblocks = []
  #n_layers is encoder layers
  # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
  # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
  eblocks.append(EncoderBlock(inputs, n_filters,dropout_prob=0, max_pooling=True))
  for i in range(n_layers-1):
    dropout = 0
    if(i > 1):
      dropout = 0.3
    eblocks.append(EncoderBlock(eblocks[-1][0], n_filters*(2**(i+1)), dropout_prob=dropout, max_pooling=True))

  #intermediate latent also put in encoder
  eblocks.append(EncoderBlock(eblocks[-1][0], n_filters*(2**(n_layers)), dropout_prob=0, max_pooling=False)) 

  dblocks = []
  dblocks.append(DecoderBlock(eblocks[-1][0], eblocks[-2][1],  n_filters*(2**(n_layers-1))))
  for i in range(n_layers-1): 
    dblocks.append(DecoderBlock(dblocks[-1], eblocks[-i-3][1],  n_filters*(2**(n_layers-2-i))))

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

def check_gpu():
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

def dice_loss(y_true, y_pred):
  
    y_pred = tf.argmax(y_pred, 3)
    y_true = y_true[:, :, :, 0]

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    dice_coef = numerator / denominator

    return 1 - dice_coef

SCCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def unet_function(X_train, y_train, X_valid, y_valid, X_test, y_test, dataset = "unspecified", 
                 loss = SCCE, optimizer=tf.keras.optimizers.Adam(),callback_type = "", callbacks = [],
                 batch_size=32, epochs=200, 
                 input_size=(128, 128, 1), n_filters=32, n_classes=2, n_layers = 4, dropout=0.3,
                 plot_masks=True, plot_history=True, print_summary=False) :  
  print( f'Epochs = {epochs} \nloss = {loss} \ncallback type = {callback_type} \nUNet_{dataset}_{n_layers}L_{n_filters}')
  unet = build_model(input_size=input_size,n_filters=n_filters, n_classes=n_classes, n_layers = n_layers, dropout = 0.3) 
  
  if(print_summary):
    unet.summary()
    tf.keras.utils.plot_model(unet, to_file="model.png", show_shapes=True, expand_nested=True)

  unet.compile(optimizer=optimizer, 
             loss=loss,
             #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             #loss=dice_loss,
              metrics=['accuracy'])
  
  #train and evaluate
  results = train_unet(unet, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test, dataset=dataset,
                            loss = loss, optimizer=optimizer, callback_type = callback_type, callbacks = callbacks,
                            n_layers=n_layers, n_filters=n_filters,
                            batch_size=batch_size, epochs=epochs)

  return results

def train_unet(unet, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                 n_layers, n_filters, dataset = "unspecified", 
                 loss = SCCE, callback_type = "", callbacks = [],
                 batch_size=32, epochs=200,
                 plot_masks=True, plot_history=True, print_summary=False):
  actual_callbacks = callbacks
  if(callback_type == "checkpoint"):
    checkpoint_filepath = f'/content/UNet_{dataset}_{n_layers}L_{n_filters}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='loss',
                                                                mode='min',
                                                                save_best_only=True)
    actual_callbacks = [model_checkpoint_callback]
  
  if(callback_type == "patience"):
    callback_patience = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=30)
    actual_callbacks = [callback_patience]
  
  with tf.device('/device:GPU:0'):
    history = unet.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=actual_callbacks)
    
  if(callback_type == "checkpoint"):
    unet.load_weights(checkpoint_filepath)

  if(plot_history):
      plt.figure()
      plt.plot(history.history['accuracy'], label='accuracy')
      plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.ylim([0, 1]) # recall accuracy is between 0 to 1
      plt.legend(loc='lower right') # specify location of the legend
      
      plt.figure()
      plt.plot(history.history['loss'], label='loss')
      plt.plot(history.history['val_loss'], label = 'val_loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.ylim([0, 1]) # recall accuracy is between 0 to 1
      plt.legend(loc='lower right') # specify location of the legend

      plt.show()
   
  eval_results = evaluate_unet(unet = unet, X_train = X_train, y_train = y_train, X_valid = X_valid, y_valid = y_valid, X_test = X_test, y_test = y_test, dataset = dataset,
                 plot_masks = plot_masks, print_summary = print_summary)
  
  results = {
          'Data': {
              'X_train': X_train, 
              'y_train': y_train, 
              'X_valid': X_valid, 
              'y_valid': y_valid, 
              'X_test': X_test, 
              'y_test': y_test
          },
          'Model': unet,
          'History': history
      }

  results.update(eval_results)
    
  return results

def evaluate_unet(unet, X_train, y_train, X_valid, y_valid, X_test, y_test, dataset = "unspecified", 
                 plot_masks = True, print_summary = False):  

  y_train_hat_ = unet.predict(X_train.reshape((X_train.shape[0],128,128,1)))
  y_test_hat_ = unet.predict(X_test.reshape((X_test.shape[0],128,128,1)))
  y_train_hat_max = tf.argmax(y_train_hat_, 3)
  y_test_hat_max = tf.argmax(y_test_hat_, 3)

  if(plot_masks):
    #plot some results
    WIDTH, HEIGHT = 20, 20
    ROW = 1 
    COL = 10
    OFFSET = 0

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(y_test[i+OFFSET][:, :, 0], cmap='gist_gray_r')
        plt.title('true mask: ' + str(i+1))

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(y_test_hat_max[i+OFFSET], cmap='gist_gray_r')
        plt.title('pred mask: ' + str(i+1))

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(y_test_hat_max[i+OFFSET].numpy() - y_test[i+OFFSET][:, :, 0], cmap='gist_gray_r')
        plt.title('mask error: ' + str(i+1))

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(X_test[i+OFFSET][:, :, 0], cmap='gist_ncar_r')
        plt.title('true picture: ' + str(i+1))

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(y_test_hat_max[i+OFFSET].numpy() * X_test[i+OFFSET][:,:,0], cmap='gist_ncar_r')
        plt.title('pred + true: ' + str(i+1))

    plt.figure(figsize=(WIDTH, HEIGHT))
    for i in range(ROW*COL):
        plt.subplot(ROW,COL,i+1)
        plt.imshow(y_test[i+OFFSET][:, :, 0] * X_test[i+OFFSET][:,:,0], cmap='gist_ncar_r')
        plt.title('true mask+true: ' + str(i+1))

    plt.show()
  
  dice_train = dice_loss(y_true=y_train, y_pred=y_train_hat_)
  dice_test = dice_loss(y_true=y_test, y_pred=y_test_hat_)

  print("train:")
  train_loss, train_acc = unet.evaluate(X_train, y_train, verbose=2)
  print(f'Dice: {dice_train}')
  print("test:")
  test_loss, test_acc = unet.evaluate(X_test, y_test, verbose=2)
  print(f'Dice: {dice_test}')

  return {
          'Data': {
              'X_train': X_train, 
              'y_train': y_train, 
              'X_valid': X_valid, 
              'y_valid': y_valid, 
              'X_test': X_test, 
              'y_test': y_test
          },
          'Model': unet,
          'Train Result': {
              'y_train_hat_': y_train_hat_,
              'train_loss': train_loss,
              'train_acc': train_acc,
              'dice_loss': dice_train
          },
          'Test Result': {
              'y_test_hat_': y_test_hat_,
              'test_loss': test_loss,
              'test_acc': test_acc,
              'dice_loss': dice_test
          }
      }
