cmake_minimum_required(VERSION 2.8.3)
project(viz_tool)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  styx_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES styx
#  CATKIN_DEPENDS dbw_mkz_msgs geometry_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

