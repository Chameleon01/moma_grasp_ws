#!/usr/bin/env python
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moma_utils.ros.panda import PandaArmClient, PandaGripperClient

def open_gripper():
    # Initialize the moveit_commander
    rospy.init_node('open_gripper_node', anonymous=True)
    gripper = PandaGripperClient()
    rospy.loginfo("Gripper setting up")
    gripper.move(width=0.0)
    gripper.move(width=0.08)


if __name__ == '__main__':
    try:
        open_gripper()
    except rospy.ROSInterruptException:
        pass
