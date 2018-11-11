######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import PIL
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_dice'
VIDEO_NAME = 'images_dice/dice_rools.mp4'
imsize = 224


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backend_resnet = models.resnet101(pretrained=False)
num_ftrs = backend_resnet.fc.in_features
backend_resnet.fc = nn.Linear(num_ftrs, 12)
#backend_resnet = backend_resnet.to(device)
backend_resnet.load_state_dict(torch.load('C:/Users/micha/Desktop/projects/dice_detector/models/backend_resnet101.pth'))
backend_resnet.eval()


loader = transforms.Compose([transforms.Resize(imsize),
    transforms.CenterCrop(imsize), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = PIL.Image.fromarray(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image#.cuda()  #assumes that you're using GPU

def get_predicted_class(image_array):
    labels = {0:'1' , 1:'10', 2:'11', 3:'12', 4:'2', 5:'3',
     6:'4', 7:'5', 8:'6', 9:'7', 10:'8', 11:'9'}

    image = image_loader(image_array)
    y_pred = backend_resnet(image)
    #print(y_pred.cpu().data.numpy().argmax(),y_pred)

    label_out = labels[y_pred.cpu().data.numpy().argmax()]

    return label_out, y_pred

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_dice','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 12
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_dice.avi', -1, 30.0, (1280,720))


while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    frame = cv2.resize(frame,(1280,720))
    
    image_for_cropping = frame.copy()
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    frame, class_list,box_list = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)

    dice_val_sum = 0 
    for idx in range(len(box_list)):
        ymin = int(box_list[idx][0])
        xmin = int(box_list[idx][1])
        ymax = int(box_list[idx][2])
        xmax = int(box_list[idx][3])
        
        #class_number = cleaned_class_list[idx]
        #print(class_number,ymin,ymax,xmin,xmax)

        crop_img = image_for_cropping[ymin:ymax, xmin:xmax]
        pred_class ,raw_pred= get_predicted_class(crop_img)
        #print(pred_class)
        #crop_path = 'C:/Users/micha/Desktop/projects/dice_detector/images/'+class_number+'/'+str(idx)+'_'+i+file_type
        #print(crop_path)
        #cv2.imwrite(crop_path,crop_img)
        dice_val_sum += int(pred_class)
        #cv2.imshow(str(pred_class), crop_img)
        #cv2.waitKey(0)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    CornerOfText = (10,50)
    fontScale              = 2
    fontColor              = (0,255,0)
    lineType               = 2

    cv2.putText(frame,'Dice value: '+str(dice_val_sum), 
    CornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)

    out.write(frame)
    # All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    #if cv2.waitKey(1) == ord('q'):
    #    break

# Clean up
out.release()
video.release()
cv2.destroyAllWindows()
