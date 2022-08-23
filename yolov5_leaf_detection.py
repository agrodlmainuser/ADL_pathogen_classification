import os
from ADL_classes import ADL_Read_XML
import xml.etree.ElementTree as ET
import shutil

%cd drive/MyDrive/Colab Notebooks
script_params = ADL_Read_XML("AgroDL_YOLOv5_leafs_detections_0000")
folder_dir = script_params.get_params("folder_dir")
yolov5_detections_root_dir = script_params.get_params("yolov5_detections_root_dir")
%cd AgroML_Test_System/yolov5
%pip install -qr requirements.txt  # install
import utils
display = utils.notebook_init()  # checks
#implementing yolov5 detection on relevant leaf detection weight
!python detect.py --weights leafs1.pt --img 640 --conf 0.25 --source {folder_dir} --save-txt
# get last created folder name
for filename in os.listdir('runs/detect'):
  pass
src = f"{yolov5_detections_root_dir}/{filename}"
dst = f"{folder_dir}/{filename}"
shutil.copytree(src,dst)
# Changing the yolov5 output dir name to a generic one as this name is constantly changing 
os.chdir(folder_dir)
os.rename(f"{filename}", "detected_images")
