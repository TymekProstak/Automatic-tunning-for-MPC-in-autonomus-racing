cmake_minimum_required(VERSION 3.7)
project(dv_control)

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
    roscpp std_msgs geometry_msgs nav_msgs tf2 tf2_ros dv_interfaces roslib
)

# ================================
#   ŚCIEŻKI DO ACADOS INSTALL/
# ================================
set(ACADOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/acados/install")
message(STATUS "ACADOS DIR = ${ACADOS_DIR}")

include_directories(
  include
  mpc_solver
  mpc_solver/acados_solver_generated
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${ACADOS_DIR}/include
  ${ACADOS_DIR}/include/acados
  ${ACADOS_DIR}/include/blasfeo/include
  ${ACADOS_DIR}/include/hpipm/include
)

link_directories(
  ${ACADOS_DIR}/lib
)

# ==============================================
#  BIBLIOTEKA SOLVERA MPC z ACADOS
# ==============================================
file(GLOB_RECURSE ACADOS_C_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/mpc_solver/acados_solver_generated/*.c
)

add_library(mpc_ltv_solver SHARED
  ${ACADOS_C_FILES}
)

target_link_libraries(mpc_ltv_solver
  ${ACADOS_DIR}/lib/libacados.so
  ${ACADOS_DIR}/lib/libhpipm.so
  ${ACADOS_DIR}/lib/libblasfeo.so
  ${ACADOS_DIR}/lib/libqpOASES_e.so
  m
)

# ======================================================
#  EXECUTABLE - ROS node
# ======================================================
set(control_SRC
  src/stanley.cpp
  src/wrapper.cpp
  src/mpc_agh_racing.cpp
)

add_executable(dv_control_node
  src/main.cpp
  ${control_SRC}
)

target_link_libraries(dv_control_node
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  mpc_ltv_solver
  m
)

# ==========================
#  RPATH (żeby nie exportować LD_LIBRARY_PATH)
# ==========================
set_target_properties(mpc_ltv_solver PROPERTIES
  BUILD_RPATH "${ACADOS_DIR}/lib"
  INSTALL_RPATH "${ACADOS_DIR}/lib"
)

set_target_properties(dv_control_node PROPERTIES
  BUILD_RPATH "${ACADOS_DIR}/lib"
  INSTALL_RPATH "${ACADOS_DIR}/lib"
)