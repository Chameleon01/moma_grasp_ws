#!/usr/bin/env python3

import sys
import numpy as np

from actionlib import SimpleActionServer, SimpleActionClient
from sensor_msgs.msg import Image, PointCloud2
import rospy
import std_srvs.srv

from grasp_demo.msg import ScanSceneAction, ScanSceneResult
from moma_utils.ros.moveit import MoveItClient
from moma_utils.spatial import Transform, Rotation
import vpp_msgs.srv
import vgn.srv
import tf

def move():
    moveit_manip = MoveItClient("panda_manipulator")
    moveit_arm = MoveItClient("panda_arm")
    init_pose = [-0.174533,-1.74533, 0.100, -2.416, 0.089, 1.402, 0.851]
    init_pose = [0.0024443426956803904, -0.44002462433115763, -0.00045557918923488927, -1.1405253834923474, -0.0020210342712445595, 2.1378287625343617, 0.7857865308860124]

    moveit_arm.goto(init_pose, velocity_scaling=0.2)
    curr_pose = np.array([moveit_manip.move_group.get_current_pose().pose.position.x, 
                          moveit_manip.move_group.get_current_pose().pose.position.y,
                          moveit_manip.move_group.get_current_pose().pose.position.z])
    curr_rot = np.array([moveit_manip.move_group.get_current_pose().pose.orientation.x,
                         moveit_manip.move_group.get_current_pose().pose.orientation.y,
                         moveit_manip.move_group.get_current_pose().pose.orientation.z,
                         moveit_manip.move_group.get_current_pose().pose.orientation.w])

    for ang in range(-90, 100, 10):
        alpha = ang*(np.pi/180) # convert highest_avg_azimuth in radians

        # translation follow an ellipse
        a = 0.15 # distance from base to object
        b = 0.15 # lateral distance
        t = np.arctan(np.tan(alpha)*a/b)
        x_trans = a*np.cos(t)
        y_trans = b*np.sin(t)
        z_trans = -0.25

        # shift the frame
        x_trans -= 0.2

        # rotation rotate around z axis of alpha radians and put in a quaternion
        q = tf.transformations.quaternion_from_euler(alpha, 0, 0)

        output_array = [-x_trans, -y_trans, z_trans, q[0], q[1], q[2], q[3]]

        temp = curr_pose+output_array[:3]
        new_rot = tf.transformations.quaternion_multiply(curr_rot,  output_array[3:]) 

        target_pose = Transform(translation=temp, rotation=Rotation.from_quat(new_rot))
        moveit_manip.goto(target_pose)
        rospy.loginfo("current pose: %s", curr_pose)
        rospy.loginfo("current rotation: %s", curr_rot)
        rospy.loginfo("target pose: %s", temp)
        rospy.loginfo("curr angle: %s", ang)
        
        rospy.sleep(0.5)

    target_pose = Transform(translation=curr_pose, rotation=Rotation.from_quat(curr_rot))
    moveit_manip.goto(target_pose)

def get_joint_values():
    moveit = MoveItClient("panda_arm")
    joint_values = moveit.move_group.get_current_joint_values()
    rospy.loginfo(joint_values)

if __name__ == '__main__':
    rospy.init_node('move_end_effector_node', anonymous=True)
    move()