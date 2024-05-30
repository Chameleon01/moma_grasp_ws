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

def move_to_viewpoint(init_trans, init_rot, ang_deg, object_pose=np.array([0.5, -0.05, 0.5])):
    moveit_manip = MoveItClient("panda_manipulator")
    alpha = ang_deg*(np.pi/180) # convert highest_avg_azimuth in radians
    radius = object_pose[0] - init_trans[0]
    # translation follow a circle
    x_trans = radius*np.cos(alpha)
    y_trans = radius*np.sin(alpha)
    z_trans = 0

    # ellipse
    # t = np.arctan(np.tan(alpha)*a/b)
    # x_trans = a*np.cos(t)
    # y_trans = b*np.sin(t)

    # shift the trajectory in x and y in order to center the object spwaned
    rospy.loginfo("curret pos: %s", moveit_manip.move_group.get_current_pose().pose.position.x)
    x_trans -= object_pose[0] - init_trans[0]
    y_trans -= object_pose[1] - init_trans[1]

    # rotation rotate around x and y axis
    # y_anglle based on radius and z distance from object
    y_ang = np.arctan((init_trans[2]-object_pose[2])/radius)
    q = tf.transformations.quaternion_from_euler(alpha, -y_ang, 0)

    trans = [-x_trans, -y_trans, z_trans]

    new_trans = init_trans+trans
    new_rot = tf.transformations.quaternion_multiply(init_rot,  q) 

    target_pose = Transform(translation=new_trans, rotation=Rotation.from_quat(new_rot))
    moveit_manip.gotoL(target_pose)
    rospy.loginfo("target pose: %s", new_trans)
    rospy.loginfo("curr angle: %s", ang_deg)
    
    rospy.sleep(0.5)

def move():
    moveit_arm = MoveItClient("panda_arm")
    init_pose = [0.0, -1.2859993464471628, -0.0007797185767710602, -2.785808430007588, -0.0021583956237938295, 3.0701282814749877, 0.7863824527452925]
    moveit_arm.goto(init_pose, velocity_scaling=0.2)

    moveit_manip = MoveItClient("panda_manipulator")

    # move end effector to initial position
    init_trans = np.array([0.3, 0.0, 0.6])
    init_ros = tf.transformations.quaternion_from_euler(0, -np.pi/2, np.pi)
    target_pose = Transform(translation=init_trans, rotation=Rotation.from_quat(init_ros))
    moveit_manip.goto(target_pose)

    curr_trans = np.array([moveit_manip.move_group.get_current_pose().pose.position.x, 
                          moveit_manip.move_group.get_current_pose().pose.position.y,
                          moveit_manip.move_group.get_current_pose().pose.position.z])
    curr_rot = np.array([moveit_manip.move_group.get_current_pose().pose.orientation.x,
                         moveit_manip.move_group.get_current_pose().pose.orientation.y,
                         moveit_manip.move_group.get_current_pose().pose.orientation.z,
                         moveit_manip.move_group.get_current_pose().pose.orientation.w])
    rospy.loginfo("current trans: %s", curr_trans)
    rospy.loginfo("current rot: %s", curr_rot)
    move_to_viewpoint(init_trans=curr_trans, init_rot=curr_rot, ang_deg=0)
    # for ang in range(-90, 100, 10):
    #     move_to_viewpoint(init_trans=curr_trans, init_rot=curr_rot, ang_deg=ang)

def move_old():
    moveit_manip = MoveItClient("panda_manipulator")
    moveit_arm = MoveItClient("panda_arm")
    init_pose = [-0.174533,-1.74533, 0.100, -2.416, 0.089, 1.402, 0.851]
    init_pose = [0.0, -1.2859993464471628, -0.0007797185767710602, -2.785808430007588, -0.0021583956237938295, 3.0701282814749877, 0.7863824527452925]

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
        z_trans = 0

        # shift the frame
        dis_to_object = 0.2
        x_trans -= 0.2

        # rotation rotate around x axis
        x_q= tf.transformations.quaternion_from_euler(alpha, 0, 0)
        # rotation rotate around y axis
        y_ang = 30*(np.pi/180)
        y_q = tf.transformations.quaternion_from_euler(0, -y_ang, 0)

        trans = [-x_trans, -y_trans, z_trans]

        temp = curr_pose+trans
        new_rot = tf.transformations.quaternion_multiply(curr_rot,  x_q) 
        new_rot = tf.transformations.quaternion_multiply(new_rot,  y_q) 

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
    get_joint_values()