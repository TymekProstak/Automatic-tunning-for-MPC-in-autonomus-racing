cmake_minimum_required(VERSION 3.12)
project(dv_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-O2)

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
  INCLUDE_DIRS include
  CATKIN_DEPENDS
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

# ================================
#  ACADOS SYSROOT
# ================================
set(ACADOS_SYSROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/acados_sysroot")
message(STATUS "ACADOS SYSROOT DIR = ${ACADOS_SYSROOT_DIR}")

# ======================================================
#  ACADOS generated solver sources (FRENET runtime)
# ======================================================
set(ACADOS_SOLVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mpc_solver/acados_solver_generated_frenet_centerline_runtime")
message(STATUS "ACADOS SOLVER DIR = ${ACADOS_SOLVER_DIR}")

file(GLOB_RECURSE ACADOS_C_FILES CONFIGURE_DEPENDS
  "${ACADOS_SOLVER_DIR}/*.c"
)

# wykluczam programy testowe / sim solver
list(FILTER ACADOS_C_FILES EXCLUDE REGEX ".*/main_.*\\.c$")
list(FILTER ACADOS_C_FILES EXCLUDE REGEX ".*/main_sim_.*\\.c$")
list(FILTER ACADOS_C_FILES EXCLUDE REGEX ".*/acados_sim_solver_.*\\.c$")

if(NOT ACADOS_C_FILES)
  message(FATAL_ERROR
    "No ACADOS solver .c files found in: ${ACADOS_SOLVER_DIR}\n"
    "Did you run the Frenet solver generator?"
  )
endif()

# ==========================
#  Common include dirs
# ==========================
set(DV_CONTROL_INCLUDE_DIRS
  include
  mpc_solver
  ${ACADOS_SOLVER_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ACADOS_SYSROOT_DIR}/include
  ${ACADOS_SYSROOT_DIR}/include/acados
  ${ACADOS_SYSROOT_DIR}/include/acados_c
  ${ACADOS_SYSROOT_DIR}/include/blasfeo/include
  ${ACADOS_SYSROOT_DIR}/include/hpipm/include
)

# ==============================================
#  Generated ACADOS solver library
# ==============================================
add_library(frenet_centerline_solver SHARED
  ${ACADOS_C_FILES}
)

target_include_directories(frenet_centerline_solver
  PUBLIC
    ${DV_CONTROL_INCLUDE_DIRS}
)

target_link_directories(frenet_centerline_solver
  PUBLIC
    ${ACADOS_SYSROOT_DIR}/lib
)

target_link_libraries(frenet_centerline_solver
  PUBLIC
    acados
    hpipm
    blasfeo
    m
)

add_dependencies(frenet_centerline_solver
  ${catkin_EXPORTED_TARGETS}
)

# ======================================================
#  ROS node
# ======================================================
set(CONTROL_SRC
  src/stanley.cpp
  src/wrapper.cpp
  src/mpc_amz_racing.cpp
)

add_executable(dv_control_node
  src/main.cpp
  ${CONTROL_SRC}
)

target_include_directories(dv_control_node
  PRIVATE
    ${DV_CONTROL_INCLUDE_DIRS}
)

target_link_directories(dv_control_node
  PRIVATE
    ${ACADOS_SYSROOT_DIR}/lib
)

target_link_libraries(dv_control_node
  PRIVATE
    ${catkin_LIBRARIES}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    frenet_centerline_solver
    m
)

add_dependencies(dv_control_node
  ${catkin_EXPORTED_TARGETS}
)

# ==========================
#  RPATH
# ==========================
set_target_properties(frenet_centerline_solver PROPERTIES
  BUILD_RPATH "${ACADOS_SYSROOT_DIR}/lib"
  INSTALL_RPATH "${ACADOS_SYSROOT_DIR}/lib"
)

set_target_properties(dv_control_node PROPERTIES
  BUILD_RPATH "${ACADOS_SYSROOT_DIR}/lib"
  INSTALL_RPATH "${ACADOS_SYSROOT_DIR}/lib"
)