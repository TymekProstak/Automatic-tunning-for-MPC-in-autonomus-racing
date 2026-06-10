cmake_minimum_required(VERSION 3.7)
project(dv_control)

# --- Kompilator ---
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
add_compile_options(-Wall -Wextra)

# ==========================
#  FIND PACKAGES ROS + JSON
# ==========================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  visualization_msgs
  dv_interfaces
  roslib
)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp std_msgs geometry_msgs nav_msgs tf2 tf2_ros visualization_msgs dv_interfaces roslib
)

# ==========================
#  INCLUDE DIRECTORIES
# ==========================
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ======================================================
#  EXECUTABLE - ROS node
# ======================================================
set(control_SRC
  src/stanley.cpp
  src/wrapper.cpp
  src/mpc_ka_racing.cpp
  # jeśli masz jeszcze inne pliki, dopisz tutaj
)

add_executable(dv_control_node
  src/main.cpp
  ${control_SRC}
)

target_link_libraries(dv_control_node
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  m
)
