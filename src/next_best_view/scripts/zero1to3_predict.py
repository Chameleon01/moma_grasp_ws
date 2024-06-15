#!/usr/bin/env python
import rospy
import cv2
import os

def load_zero1to3_predictions(n_img=1):
    # Path to the images directory
    images_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'img_predictions')
    img_predictions = []
    for az in [-80, -60, -40, -20, 0, 20, 40, 60, 80]:
        file_path = f"{images_dir}/output{n_img}.{az}.png"
        rospy.loginfo(f"Loading image: {file_path}")
        image = cv2.imread(file_path)
        if image is not None:
            pred_obj = {"image": image, "azimuth": az}
            img_predictions.append(pred_obj)

    return img_predictions

def get_img_keys(selected=False):
    # get the file name list in the directory img_predictions 
    # [3, 7, 8, 11, 18, 19,20, 28,32, 41, 44, 45, 48, 49, 50],[49] mug
    if selected:
        return [11] 
    file_names = os.listdir('/root/moma_ws/src/next_best_view/scripts/img_predictions')
    file_names = [f.split(".") for f in file_names]
    file_names = [f[0].replace("output", "") for f in file_names]
    file_names = list(set(file_names))
    file_names = [int(f) for f in file_names]
    file_names.sort()
    
    return file_names
    

