# pathogen_classification
This repo contains all scripts associated to disease classification

Pathogen detection via YOLOv5:

- train.py - training the model based on a TL pre-trained model. Saving the weights at the end in a dedicated directory. 
- yolov5_leaf_detection.py - perform leaves detection to an image from the GH to be used later. 
- Leaves_isolation.py - isolate the detected leaves from the script above
- test.py - test an image from the field and creates bounding boxes marks around every detection. 

Pathogen detection via Segmentation:
