#!/usr/bin/env python
import rospy
from information_gain.msg import zero1to3Image, zero1to3Images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

def load_zero1to3_predictions():
    # Path to the images directory
    images_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'img_predictions')
    n_img = 1
    img_predictions = []
    for az in range(0, 360, 45):
        file_path = f"{images_dir}/output{n_img}.{az}.png"
        rospy.loginfo(f"Loading image: {file_path}")
        image = cv2.imread(file_path)
        if image is not None:
            pred_obj = {"image": image, "azimuth": az}
            img_predictions.append(pred_obj)

    return img_predictions
