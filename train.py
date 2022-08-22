import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
from ADL_classes import ADL_Read_XML
import xml.etree.ElementTree as ET

# parsing parameters from XML file
script_parameters = ADL_Read_XML("AgroDL_Classification_TL_Training_0001")
input_dir = script_parameters.get_params("original_input_dir")
val_dir = script_parameters.get_params("original_val_dir")
classification_model_save_root = script_parameters.get_params("classification_model_save_root")


# hyper parameters value setting
BATCH_SIZE = 32
CHANNELS=3
EPOCHS=15


for class_dir in os.listdir(train_dir):
  class_val_dir = f'{val_dir}/{class_dir}'
  if os.path.exists(class_val_dir):
    continue 
  else:
    os.mkdir(class_val_dir)
    for i,image in enumerate(os.listdir(f'{train_dir}/{class_dir}')):
      if i%5 == 0:
        shutil.move(f'{train_dir}/{class_dir}/{image}', f'{class_val_dir}/{image}')


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    batch_size=BATCH_SIZE)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
n_classes = len(class_names)


class Ptgn_Classification:
  
  def __init__(self, BATCH_SIZE=32, CHANNELS=3, EPOCHS=15, train_dir, val_dir, classification_model_save_root, n_classes):
    self.BATCH_SIZE = BATCH_SIZE
    self.CHANNELS = CHANNELS
    self.EPOCHS = EPOCHS
    self.train_dir = train_dir
    self.val_dir = val_dir
    self.classification_model_save_root = classification_model_save_root
    self.n_classes = n_classes

    
  def prepare_ds(self):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(self.train_dir, batch_size=self.BATCH_SIZE)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(self.val_dir,batch_size=self.BATCH_SIZE)
    return train_ds, val_ds
  
  def train(self, tl_model_name_str):
    self.tl_model_name_str = tl_model_name_str
    # pulling the tl models from tf hub
    if self.tl_model_name_str == "mobilenet_v3":
      tl_model = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    elif self.tl_model_name_str == "efficientnet_v2":
      tl_model = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_b2/feature_vector/2"
    elif self.tl_model_name_str == "inception_v3":
      tl_model = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
    # activating previous function to get the ds at the right struction
    train_ds, val_ds = self.prepare_ds()
    resize_and_rescale = tf.keras.Sequential([layers.experimental.preprocessing.Resizing(self.IMAGE_SIZE, self.IMAGE_SIZE),
                                              layers.experimental.preprocessing.Rescaling(1./255),])
    data_augmentation = tf.keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                                             layers.experimental.preprocessing.RandomRotation(0.2),])
    pretrained_model_without_top_layer = hub.KerasLayer(tl_model, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3), trainable=False)
    
    model = tf.keras.Sequential([resize_and_rescale, data_augmentation, pretrained_model_without_top_layer,
                               tf.keras.layers.Dense(256, activation='relu'),
                               tf.keras.layers.Dense(n_classes, activation='softmax')])
    model.build([None, self.IMAGE_SIZE, self.IMAGE_SIZE, 3]) 
    model.summary()
		model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    history = model.fit(train_ds, batch_size=self.BATCH_SIZE, validation_data=val_ds, verbose=1,epochs=self.EPOCHS,)
    # for visualing the model
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # draw a graph
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(self.EPOCHS), acc, label='Training Accuracy')
    plt.plot(range(self.EPOCHS), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(self.EPOCHS), loss, label='Training Loss')
    plt.plot(range(self.EPOCHS), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.figure(figsize=(20, 50))

  def model_save(self, blob=False):
    # here a new output directory is created for the saved model
    for i in range(1000):
      if blob:
        extn = f"exp{i}_{self.tl_model_name_str}_blob"
      else:
        extn = f"exp{i}_{self.tl_model_name_str}"
      if os.path.isdir(f"{self.classification_model_save_root}/{extn}"):
        continue
      else:
        os.mkdir(f"{self.classification_model_save_root}/{extn}")
        break
    model.save(f"{self.classification_model_save_root}/{extn}")





# Retrain an existed model
classification_model_save_root = "/content/drive/MyDrive/Colab Notebooks/AgroML_Test_System/Clasification_Models/"
tl_model_str = "mobilenet_v3"
trained_model = keras.models.load_model(f'{model_save_root}/exp0_{tl_model_str}_{EPOCHS}epochs_224img_size')
trained_model.summary()

def retrain_net(trained_model_input,IMAGE_SIZE, tl_model_str):

  trained_model.summary()

  history = trained_model.fit(train_ds, batch_size=BATCH_SIZE,
                       validation_data=val_ds, verbose=1,
                      epochs=EPOCHS,)
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  # here a new output directory is created for the saved trained_model
  model_save_root = "/content/drive/MyDrive/Colab Notebooks/AgroML_Test_System/Clasification_Models/"
  for i in range(1000):
    if os.path.isdir(f"{model_save_root}/exp{i}_{tl_model_str}_retrain"):
      continue
    else:
      os.mkdir(f"{model_save_root}/exp{i}_{tl_model_str}_retrain")
      break
  trained_model.save(f"{model_save_root}/exp{i}_{tl_model_str}_retrain")
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(range(EPOCHS), acc, label='Training Accuracy')
  plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(range(EPOCHS), loss, label='Training Loss')
  plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()
  plt.figure(figsize=(20, 50))
  
  return trained_model

# the execution
retrained_model = retrain_net(trained_model,224, tl_model_str)
 




    
 




    
