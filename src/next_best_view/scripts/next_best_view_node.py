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

from moma_utils.ros.moveit import MoveItClient
from moma_utils.spatial import Transform, Rotation
import random


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
        self.img_count = 0
        self.img_keys = get_img_keys(selected=True)
        self.idx_publisher = rospy.Publisher('/spawn_model_idx', Float32MultiArray)
        rospy.loginfo(f'Image keys: {self.img_keys}')

        self.moveit_manip = MoveItClient("panda_manipulator")
        self.moveit_arm = MoveItClient("panda_arm")
        self.init_trans = np.array([0.3, 0.0, 0.6])
        self.init_rot = tf.transformations.quaternion_from_euler(0, -np.pi/2, np.pi)
        self.object_pose=np.array([0.5, 0.0, 0.5])

        self.curr_mode = 0 # "best": 0, "static": 1, "random": 2
        self.run_num = 5

    def init_pose(self):
        init_pose = [0.0, -1.2859993464471628, -0.0007797185767710602, -2.785808430007588, -0.0021583956237938295, 3.0701282814749877, 0.7863824527452925]
        init_pose = [1.0099570194674232, -1.692222709995713, -0.39451387234134483, -2.676419590979598, -1.2053577622836915, 1.8697722273445159, 1.4881047141400767]
        self.moveit_arm.goto(init_pose, velocity_scaling=0.2)

        self.move_to_view(0)

    def move_to_view(self, ang_deg):
        alpha = ang_deg*(np.pi/180) # convert highest_avg_azimuth in radians
        radius = self.object_pose[0] - self.init_trans[0]
        # translation follow a circle
        x_trans = radius*np.cos(alpha)
        y_trans = radius*np.sin(alpha)
        z_trans = 0

        # ellipse
        # t = np.arctan(np.tan(alpha)*a/b)
        # x_trans = a*np.cos(t)
        # y_trans = b*np.sin(t)

        # shift the trajectory in x and y in order to center the object spwaned
        x_trans -= self.object_pose[0] - self.init_trans[0]
        y_trans -= self.object_pose[1] - self.init_trans[1]

        # rotation rotate around x and y axis
        # y_anglle based on radius and z distance from object
        offset = 10 * (np.pi/180)
        if abs(ang_deg) >= 80:
            offset *= 4
        y_ang = np.arctan((self.init_trans[2]-self.object_pose[2])/radius) + offset
        q = tf.transformations.quaternion_from_euler(alpha, -y_ang, 0)

        trans = [-x_trans, -y_trans, z_trans]

        new_trans = self.init_trans+trans
        new_rot = tf.transformations.quaternion_multiply(self.init_rot,  q) 
        # new_rot = tf.transformations.quaternion_multiply(new_rot,  tf.transformations.quaternion_from_euler(0, 0, np.pi/4)) 

        target_pose = Transform(translation=new_trans, rotation=Rotation.from_quat(new_rot))
        self.moveit_manip.goto(target_pose)
        
        rospy.sleep(0.5)

    def execute(self, goal):
        # -- for evaluation purposes --
        if self.img_count >= len(self.img_keys):
            self.img_count = 0
            self.curr_mode += 1
            rospy.loginfo(f'Current mode: {self.curr_mode}')
        if self.curr_mode > 2:
            rospy.loginfo('All modes have been executed, proceeding next run.')
            self.curr_mode = 0
            self.run_num += 1
            return
        
        spawn_msg = Float32MultiArray()
        spawn_msg.data = [self.img_keys[self.img_count], self.curr_mode, self.run_num]
        self.idx_publisher.publish(spawn_msg)
        rospy.sleep(2.0) # wait for spawn the model
        # -- ---------------------- --
    
        # call zero 1 to 3 node if available else load pre-saved predictions
        # convert image to cv2
        img = goal.image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        predictions = load_zero1to3_predictions(n_img=self.img_keys[self.img_count])

        # move the robot in initial position
        self.init_pose()

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
            if az in [-80, 80]:
                continue

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
            img_filename = os.path.join("temp", f'/root/moma_ws/src/temp/img_{self.img_keys[self.img_count]}.{az}.png')
            cv2.imwrite(img_filename, draw)
            
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

        if self.curr_mode == 0:
            self.move_to_view(highest_avg_azimuth)
        elif self.curr_mode == 1:
            self.move_to_view(0)    
        elif self.curr_mode == 2:
            self.move_to_view(random.randint(-90, 90))


        rospy.sleep(1.0)

        # Set the result of the action, not needed for now
        output_array = [0]

        # Create the result message
        result = GetNextBestViewResult()
        result.output.data = output_array 
        self.server.set_succeeded(result)

        self.img_count += 1

if __name__ == '__main__':
    rospy.init_node('next_best_view_node')
    processor = NextBestView()
    rospy.spin()
