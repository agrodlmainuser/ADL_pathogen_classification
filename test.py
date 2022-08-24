import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shutil
import os
import cv2
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import re
from ADL_classes import ADL_Read_XML
import xml.etree.ElementTree as ET



script_params = ADL_Read_XML("AgroDL_Original_Leaf_Classification_Detections_0001")
trained_model_name = script_params.get_params("trained_model_name")
trained_model_dir = script_params.get_params("trained_model_dir")
train_dir = script_params.get_params("train_dir")
images_dir = script_params.get_params("images_dir")
original_img_dir_root = script_params.get_params("original_img_dir_root")
output_root = script_params.get_params("output_root")
image_output_path = script_params.get_params("image_output_path")

trained_model = keras.models.load_model(f"{trained_model_dir}/{trained_model_name}")
trained_model.summary()
class_names = []
for class_name in os.listdir(train_dir):
  class_names.append(class_name)

  
class TestPathogen:
  
  
  def __init__(self, trained_model_dir):
    self.trained_model_dir = trained_model_dir
    
  def  load_model(self, model_name):
    self.current_model = keras.models.load_model(f"{trained_model_dir}/{model_name}")
    
  def test_classification(self, IMAGE_SIZE):
    i = 0 
    final_output = []
    # the test_images_dir contains seperated forlders for every original image from the field and therefore
    # in every of these folders there are leaves images
    for leaf_folder in os.listdir(images_dir):
      output_list = []
      conf_list = []
      class_list = []
      filenames_list = []
      for leaf_image in os.listdir(f"{images_dir}/{leaf_folder}"):
        leaf_image_path = f'{images_dir}/{leaf_folder}/{leaf_image}'
        new_img = tf.keras.preprocessing.image.load_img(leaf_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = tf.keras.preprocessing.image.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        prediction = self.current_model.predict(img)
        d = prediction.flatten()
        j = d.max()
        for index,item in enumerate(d):
          if item == j:
              class_name = class_names[index]
        confidence = round(100 * j, 3)
        conf_list.append(confidence)
        class_list.append(class_name)
        filenames_list.append(leaf_image)
      output_list.append(filenames_list)
      output_list.append(class_list)
      output_list.append(conf_list)
      output_list.append(leaf_folder)
      final_output.append(output_list)

    self.leaf_results = final_output
  
  def create_bbxs_and_csv(self):
    
    """
    The output that is going to be created contain the original image from the field with
    bbxs around all detected leaves with a negative detection (i.e. not healthy) + 
    a csv. file that conclude these detection for every image seperately
    """
    for i, image in enumerate(os.listdir(images_dir)):
      if not image.endswith("jpg"):
        continue
      else:
        for image_detections in self.leaf_results:
          if not image_detections[3] == image:
            continue
          else:
            original_img_dir = f"{original_img_dir_root}/{image}"
            img_to_be_shown = cv2.imread(f"{original_img_dir}")
            image_name_only = os.path.splitext(image)[0]
            labels = pd.read_csv(f"{output_root}/{image_name_only}.txt", sep = " ", header = None)
            labels['leaf_class'] = image_detections[1]
            labels['detection_confidence'] = image_detections[2]
            # read input image from your computer - needed for torch 
            img = read_image(f"{original_img_dir}")
            b_boxes = []
            b_box_labels = []
            detection_indexes = [] 
            for i in range(len(labels)):
              if re.findall('healthy', labels['leaf_class'][i]) or re.findall('NonPlant', labels['leaf_class'][i]): 
                continue
              else: 
                detection_indexes.append(i) 
                x_center = labels[1][i]
                y_center = labels[2][i]
                x_l = labels[3][i]*img_to_be_shown.shape[1]
                y_l = labels[4][i]*img_to_be_shown.shape[0]
                x_c_img = x_center*(img_to_be_shown.shape[1])
                y_c_img = y_center*(img_to_be_shown.shape[0])
                x_start = int(x_c_img - x_l/2)
                x_end = int(x_c_img + x_l/2)
                y_start = int(y_c_img - y_l/2)
                y_end = int(y_c_img + y_l/2)
                # bounding box are xmin, ymin, xmax, ymax
                current_box = [x_start, y_start, x_end, y_end]
                b_boxes.append(current_box)
                b_box_labels.append(labels['leaf_class'][i] + " " + str(labels['detection_confidence'][i])+"%")
            b_boxes = torch.tensor(b_boxes, dtype=torch.int)
            # draw bounding box and fill color
            img = draw_bounding_boxes(img, b_boxes, width=5,
                                      colors="white",
                                      labels = b_box_labels ,
                                      fill=False)
            # transform this image to PIL image
            img = torchvision.transforms.ToPILImage()(img)
            # save image to output
            img.save(f"{image_output_path}/{image}")
            # create corresponding csv file
            labels.drop([0,3,4], inplace=True, axis=1)
            labels = labels.rename(columns={1:"X", 2:"Y"})
            labels = labels.loc[detection_indexes]
            labels = labels.reset_index(drop=True)
            labels.to_csv(f'{image_output_path}/{image_name_only}.csv', index=None) 
            labels.to_csv("ffff.csv",index=None)  
            print(labels) 

  def delete_sub_input_dirs(self):
    # delete sub folders from input
    shutil.rmtree(f"{original_img_dir_root}/detected_images")
    shutil.rmtree(f"{original_img_dir_root}/extracted_leafs")
    

    
test_object = TestPathogen(trained_model_dir)
test_object.load_model("exp0_mobilenet_v3_retrain")
test_object.test_classification(224)
test_object.create_bbxs_and_csv()
