cmake_minimum_required(VERSION 3.0.2)
project(dv_interfaces)

find_package(catkin REQUIRED
  COMPONENTS
    actionlib_msgs
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation
)

add_message_files(
  FILES
  Imu.msg
  BoundingBox.msg
  BoundingBoxes.msg
  Cone.msg
  Cones.msg
  Control.msg
  DataLogger.msg
  DV_board.msg
  Pos.msg
  Monitoring.msg
  ObjectCount.msg
  Odometry.msg
  Pose2dStamped.msg
  ControlCommandSim.msg
  Path.msg
  Stanley_debug.msg
  full_state.msg
  MPCDebug.msg
  LtoResult.msg
  RackAngleSensor.msg
 
)

add_service_files(
  FILES
  #Uint8.srv
  SlamReset.srv
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  actionlib_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
