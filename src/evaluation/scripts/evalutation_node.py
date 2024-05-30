#!/usr/bin/env python

import rospy
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
import rospkg
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
from time import sleep
from std_msgs.msg import Int32  # Import Int32 message type
import numpy as np
import ros_numpy

class ModelSpawner:
    def __init__(self):
        rospy.init_node('evaluation')
        self.rospack = rospkg.RosPack()
        self.package_path = self.rospack.get_path('moma_gazebo')  # Adjust the package name as needed
        self.model_dir_path = f"{self.package_path}/models"
        self.pose = Pose()
        self.pose.position.x = 0.5
        self.pose.position.y = 0
        self.pose.position.z = 0.5
        self.last_model = ""
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/wrist_camera/color/image_raw", Image, self.image_callback)
        self.int_sub = rospy.Subscriber("/spawn_model_idx", Int32, self.int_callback)  # Subscriber for integer messages
        self.int_sub = rospy.Subscriber("/quality", PointCloud2, self.analyse_quality)  # Subscriber for integer messages
        self.rate = rospy.Rate(0.5)  # 0.5 Hz, or every 2 seconds
        self.img_capture = None
        self.new_image_received = False
        self.curr_model_idx = -1

    def int_callback(self, msg):
        """ Callback function for handling incoming integer messages """
        received_idx = msg.data
        if received_idx != self.curr_model_idx:
            self.curr_model_idx = received_idx
            model_names = self.list_model_directories()
            for model_name in model_names:
                self.delete_model(model_name)

            model_name = model_names[self.curr_model_idx]
            model_name_sdf = model_name[4:]  # Adjust according to your model naming convention
            model_path = f"{self.package_path}/models/{model_name}/{model_name_sdf}.sdf"
            rospy.loginfo(model_path)
            self.spawn_model(model_name, model_path)

            sleep(2)
            # Wait for a new image to ensure it corresponds to the current model
            while not self.new_image_received:
                self.rate.sleep()
            cv2.imwrite(f"img_raw/img{received_idx}.png", self.img_capture)
            self.new_image_received = False  # Reset the flag after saving the image
            self.rate.sleep()
            

        rospy.loginfo(f"Received integer: {received_idx}")

    def list_model_directories(self):
        """ Returns a list of directories in the specified path """
        return [name for name in os.listdir(self.model_dir_path) if os.path.isdir(os.path.join(self.model_dir_path, name))]

    def spawn_model(self, model_name, model_path):
        """ Spawns a model in Gazebo """
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            with open(model_path, 'r') as model_file:
                model_xml = model_file.read()
            resp = spawn_model_service(model_name, model_xml, "", self.pose, "world")
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("Model spawn failed: %s" % e)
            return False

    def delete_model(self, model_name):
        """ Deletes a model from Gazebo """
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp = delete_model_service(model_name)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr("Model deletion failed: %s" % e)
            return False

    def image_callback(self, msg):
        """ Callback function for handling incoming images """
        self.img_capture = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.new_image_received = True

    def run(self):
        model_names = self.list_model_directories()
        for model_name in model_names:
            self.delete_model(model_name)

        for i, model_name in enumerate(model_names):
            model_name_sdf = model_name[4:]  # Adjust according to your model naming convention
            model_path = f"{self.package_path}/models/{model_name}/{model_name_sdf}.sdf"
            rospy.loginfo(model_path)
            if self.spawn_model(model_name, model_path):
                if self.last_model:
                    self.delete_model(self.last_model)
                self.last_model = model_name

            sleep(2)
            # Wait for a new image to ensure it corresponds to the current model
            while not self.new_image_received:
                self.rate.sleep()
            cv2.imwrite(f"img_raw/img{i}.png", self.img_capture)
            self.new_image_received = False  # Reset the flag after saving the image
            self.rate.sleep()

    def analyse_quality(self, msg):
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        qualities = []
        for vox in pc_array:
            qualities.append(vox[3])

        qualities = np.array(qualities)
        rospy.loginfo(f"quality min: {np.min(qualities)}, quality max: {np.max(qualities)}")
        # convert msg in numpy array



if __name__ == '__main__':
    spawner = ModelSpawner()
    # spawner.run()
    rospy.spin()  # Keep the node running until terminated
