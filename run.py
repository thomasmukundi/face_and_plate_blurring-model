'''
Start by cloning the original yolo model from
https://github.com/ultralytics/yolov5

You will need to setup your base directory to be your working directory
Comment out the line: 
from google.colab.patches import cv2_imshow
Replace all cv2_imshow with cv.imshow method
You can choose the image to work with and you can select which one to blur by writing a number 
I will be working on the GUI and also a functionality that will enable you select multiple images to blur
Note that the model has not been trained on massive data and could not identify all objects in an image, 
However given a simpla image it will be able to detect well

Copy the weights file best.pt to your preferred directory and use the path for the weights
'''

import cv2
import os
import glob
import subprocess
import numpy as np
from google.colab.patches import cv2_imshow

# defining your directories
base_dir = "/content/drive/MyDrive/yolov5"
weights_path = os.path.join(base_dir, "runs/train/exp3/weights/best.pt")
 

def get_coordinates(txt_path):
    boxes = []
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    img = cv2.imread(txt_path.replace("labels", "").replace(".txt", ".jpg")) 
    img_height, img_width, _ = img.shape
    
    for line in lines:
        _, x, y, w, h = map(float, line.split())
        x, y, w, h = map(int, (x * img_width, y * img_height, w * img_width, h * img_height)) # assuming the coordinates are normalized
        boxes.append(((x-w//2, y-h//2), (x+w//2, y+h//2))) # converting center coordinates to corner coordinates
        
    return boxes


def show_image_with_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    
    for i, box in enumerate(boxes):
        cv2.putText(image, str(i), (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(image, box[0], box[1], (0, 255, 0), 2)
        
    cv2_imshow(image)
    cv2.waitKey(0)


def detect_objects(img_path):
    cmd = ["python", "/content/drive/MyDrive/yolov5/detect.py", 
           "--weights", "/content/drive/MyDrive/yolov5/runs/train/exp3/weights/best.pt", 
           "--img", "640", 
           "--conf", "0.4", 
           "--source", img_path, 
           "--save-txt"]
    subprocess.run(cmd, check=True)


def get_latest_exp_path():
    base_path = '/content/drive/MyDrive/yolov5/runs/detect'
    exp_folders = [f.name for f in os.scandir(base_path) if f.is_dir() and 'exp' in f.name]
    
    # Add filtering to only include folders that have a number after 'exp'
    exp_folders = [folder for folder in exp_folders if folder.replace('exp', '').isdigit()]

    # Ensure there's at least one experiment folder
    if not exp_folders:
        raise ValueError("No valid experiment folders found in {}".format(base_path))

    latest_exp_folder = max(exp_folders, key=lambda x: int(x.replace('exp', '')))
    return os.path.join(base_path, latest_exp_folder)



def run(img_path, blur_index=None):
    detect_objects(img_path)
    exp_path = get_latest_exp_path()
    txt_path = os.path.join(exp_path, "labels", os.path.basename(img_path).replace(".jpg", ".txt"))
    boxes = get_coordinates(txt_path)
    output_img_path = os.path.join(exp_path, os.path.basename(img_path))
    img = cv2.imread(output_img_path)
    blur_all = blur_index is None  # Check if blur_index is not provided
    
    for i, box in enumerate(boxes):
        if blur_all or (blur_index is not None and i == blur_index):
            blurred_region = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            # Increase the kernel size and sigma for stronger blurring effect
            blurred_region = cv2.GaussianBlur(blurred_region, (201, 201), 0)
            
            img[box[0][1]:box[1][1], box[0][0]:box[1][0]] = blurred_region
        
        cv2.rectangle(img, box[0], box[1], (0, 255, 0), 2)
        cv2.putText(img, str(i), (box[0][0], box[1][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    cv2_imshow(img)





image_path = "/content/drive/MyDrive/yolo2/yolo/images/train/clip16_25.jpg"
run(image_path, 1)

