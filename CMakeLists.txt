cmake_minimum_required(VERSION 3.0.2)
project(lidar_lane_detector)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
)

add_message_files(
  FILES
  CentroidWithLabelArray.msg
  CentroidWithLabel.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs sensor_msgs visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
