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
  FILES GetNextBestView.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  # INCLUDE_DIRS include  # Uncomment and fill if you have include directories
  # LIBRARIES ${PROJECT_NAME}  # Uncomment and fill if you compile libraries
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs actionlib actionlib_msgs
  # DEPENDS system_lib  # Uncomment and fill if you depend on non-ROS system libraries
)