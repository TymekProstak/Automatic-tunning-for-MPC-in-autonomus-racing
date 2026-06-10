# dv_interfaces

## Brief description
Package contains our custom driverless ROS messages.

## List of msgs:
- BoundingBox.msg
- BoundingBoxes.msg
- Cone.msg
- Cones.msg
- Control.msg
- DataLogger.msg
- DV_board.msg
- Monitoring.msg
- ObjectCount.msg
- Odometry.msg
- Pose2dStamped.msg
- ControlCommandSim.msg

### Creating a new message

1. Go to msg folder and add file with .msg extension.
2. Open CMakeLinst.txt and add you file in such block of code:

``` sh
add_message_files(
  FILES
  example.msg
)
```
If your msg use msg from other ros package add dependensies below: 
``` sh
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  message_generation
)
```
and also:
``` sh
 generate_messages(
   DEPENDENCIES
   std_msgs
)
```

Test:
1. Build your ros package - catkin_make.
2. Source devel/setup.bash and type: rosmsg show *your_pkg/your_msg* 

More information on: http://wiki.ros.org/msg
