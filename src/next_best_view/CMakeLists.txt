cmake_minimum_required(VERSION 3.0.2)
project(next_best_view)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  actionlib
  actionlib_msgs
)

add_action_files(
  DIRECTORY action
  FILES get_next_best_view.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  sensor_msgs
  std_msgs
)
