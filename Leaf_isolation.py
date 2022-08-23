import cv2
import matplotlib.pyplot as plt
import pandas as pd
from google.colab.patches import cv2_imshow
import os
from PIL import Image
from ADL_classes import ADL_Read_XML
import xml.etree.ElementTree as ET


def ceate_leafs_list(img, img_label):
  leafs_list = []
  for i in range(len(img_label)):
    x_center = img_label[1][i]
    y_center = img_label[2][i]
    x_l = img_label[3][i]*img.shape[1]
    y_l = img_label[4][i]*img.shape[0]
    x_c_img = x_center*(img.shape[1])
    y_c_img = y_center*(img.shape[0])
    x_start = int(x_c_img - x_l/2)
    x_end = int(x_c_img + x_l/2)
    y_start = int(y_c_img - y_l/2)
    y_end = int(y_c_img + y_l/2)
    crop_img = img[y_start:y_end, x_start:x_end]
    leafs_list.append(crop_img)
  return leafs_list 

%cd /content/drive/MyDrive/Colab Notebooks
for file in os.listdir(root_labeled_images_dir):
  if file.endswith("jpg"):
    img = cv2.imread(f"{root_labeled_images_dir}/{file}")
    original_image = cv2.imread(f"{input_images_dir}/{file}")
    image_name_only = os.path.splitext(file)[0]
    labels = pd.read_csv(f"{root_labeled_images_dir}/labels/{image_name_only}.txt", sep = " ", header = None)
    # Changing color channels 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # getting the coordinates list of the bbxs
    leaf_imgs = ceate_leafs_list(original_image, labels)
    for i in range(len(leaf_imgs)):
      current_img = leaf_imgs[i]
      # save the extracted leaf
      if os.path.isdir(f"{input_images_dir}/extracted_leafs"):
        pass
      else:
        os.mkdir(f"{input_images_dir}/extracted_leafs")
      if os.path.isdir(f"{input_images_dir}/extracted_leafs/{file}"):
        pass
      else:
        os.mkdir(f"{input_images_dir}/extracted_leafs/{file}")
      Image.fromarray(current_img).convert("RGB").save(f"{input_images_dir}/extracted_leafs/{file}/l{i}.jpg")
  else:
    continue
