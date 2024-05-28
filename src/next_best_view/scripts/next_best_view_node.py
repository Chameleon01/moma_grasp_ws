#!/usr/bin/env python
import rospy
import actionlib
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from zero1to3_predict import load_zero1to3_predictions, get_img_keys
from info_gain_utility import *
import os
from next_best_view.msg import GetNextBestViewAction, GetNextBestViewGoal, GetNextBestViewResult
import tf
from std_msgs.msg import Int32

# min max scaling of metrics
def min_max_scale(metrics):
    min_val = min(metrics)
    max_val = max(metrics)
    return [(x - min_val) / (max_val - min_val) for x in metrics]

class NextBestView(object):
    def __init__(self):
        self.server = actionlib.SimpleActionServer('get_next_best_view', GetNextBestViewAction, self.execute, False)
        self.server.start()
        self.bridge = CvBridge()
        self.img_count = 8
        self.img_keys = get_img_keys()
        self.idx_publisher = rospy.Publisher('/spawn_model_idx', Int32, queue_size=10)
        rospy.loginfo(f'Image keys: {self.img_keys}')

    def execute(self, goal):
        self.idx_publisher.publish(self.img_keys[self.img_count])
        predictions = load_zero1to3_predictions(n_img=self.img_keys[self.img_count])

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

        # Create the output array
        # compute translation and rotation
        alpha = highest_avg_azimuth*(np.pi/180) # convert highest_avg_azimuth in radians
        alpha = -np.pi/2# 45 degrees

        # translation follow an ellipse
        a = 0.15 # distance from base to object
        b = 0.15 # lateral distance
        t = np.arctan(np.tan(alpha)*a/b)
        x_trans = a*np.cos(t)
        y_trans = b*np.sin(t)
        z_trans = 0

        # shift the frame
        x_trans -= 0.2

        # rotation rotate around z axis of alpha radians and put in a quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, alpha)

        output_array = [-x_trans, -y_trans, z_trans, q[0], q[1], q[2], q[3]]

        # Create the result message
        result = GetNextBestViewResult()
        result.output.data = output_array  # Ensure this matches the name in your action definition
        rospy.loginfo(f"Processing complete, sending result...{output_array}")

        # Set the result of the action
        self.server.set_succeeded(result)

        self.img_count += 1

if __name__ == '__main__':
    rospy.init_node('next_best_view_node')
    processor = NextBestView()
    rospy.spin()
