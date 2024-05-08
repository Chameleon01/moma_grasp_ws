#!/usr/bin/env python
import rospy
import actionlib
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from zero1to3_predict import load_zero1to3_predictions
from info_gain_utility import *
import os


# min max scaling of metrics
def min_max_scale(metrics):
    min_val = min(metrics)
    max_val = max(metrics)
    return [(x - min_val) / (max_val - min_val) for x in metrics]

class NextBestView(object):
    def __init__(self):
        self.server = actionlib.SimpleActionServer('get_next_best_view', Image, self.execute, False)
        self.server.start()
        self.bridge = CvBridge()

    def execute(self, goal):
        
        captured_image = self.bridge.imgmsg_to_cv2(goal, "bgr8")
        # pass imgae to zero 1 to 3
        if captured_image is not None:
            cv2.imwrite("captured_image.png", captured_image)
        predictions = load_zero1to3_predictions()

        # apply gain information function to each image
        ratios = []
        entropies = []
        edge_counts = []
        corner_counts = []
        azimuths = []

        # Process each image in the array
        for index in range(len(predictions)):
            # Convert ROS Image message to an OpenCV image
            cv_image = predictions[index]['image']
            az = predictions[index]['azimuth']

            # get metrics
            src_gray = preprocess_image(cv_image)
            draw, ratio = find_and_draw_contours(src_gray)
            entropy = calculate_entropy(src_gray)
            edge_count = count_edges(src_gray)
            corner_count = count_corners(src_gray)

            ratios.append(ratio)
            entropies.append(entropy)
            edge_counts.append(edge_count)
            corner_counts.append(corner_count)  
            azimuths.append(az)

            # Construct a filename and save the image
            img_filename = os.path.join("temp", f'image_{index}.jpg')
            cv2.imwrite(img_filename, draw)
            rospy.loginfo(f'Saved {img_filename}')
            
        # min max scale metrics
        ratios = min_max_scale(ratios)
        entropies = min_max_scale(entropies)
        edge_counts = min_max_scale(edge_counts)
        corner_counts = min_max_scale(corner_counts)
        avg_metrics = [(ratios[i] + entropies[i] + edge_counts[i] + corner_counts[i]) / 4 for i in range(len(ratios))]

        # get the highest avg metric index
        highest_avg_index = avg_metrics.index(max(avg_metrics))
        # get the azimuth of the highest avg metric
        highest_avg_azimuth = azimuths[highest_avg_index]
        rospy.loginfo(f'Highest average metric: {max(avg_metrics):.2f} at azimuth: {highest_avg_azimuth:.2f}')

        output_array = [highest_avg_azimuth, 0, 0]

        # Create the result message
        result = Float32MultiArray()  # Create a result instance
        result.data = output_array  # Assign the output array to the result message
        rospy.loginfo("Processing complete, sending result...")

        # Set the result of the action
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('next_best_view_node')
    processor = NextBestView()
    rospy.spin()
