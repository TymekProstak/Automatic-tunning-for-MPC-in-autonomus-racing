cmake_minimum_required(VERSION 3.7)
project(lem_simulator)

# --- Ustawienia kompilatora ---
add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# --- ROS pakiety ---
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf
  dv_interfaces
)

# --- Zewnętrzne biblioteki ---
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

# --- catkin_package ---
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs visualization_msgs tf dv_interfaces
)

# --- Include directories ---
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/src/dv_control/include
)

# --- Wspólne źródła pomocnicze (BEZ lto.cpp) ---
set(simSRC
  src_helpers/cone_detector.cpp
  src_helpers/cone_track.cpp
  src_helpers/double_track.cpp
  src_helpers/steering_system.cpp
  src_helpers/tire_model.cpp
  src_helpers/utiliities.cpp
  src_helpers/kalman.cpp
)

# =====================================================================
#                     WĘZEŁ GŁÓWNY: SIM LOOP NODE
# =====================================================================
add_executable(sim_loop_node
  sim_loop.cpp
  main.cpp
  ${simSRC}
)

target_link_libraries(sim_loop_node PRIVATE
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  m
)

# =====================================================================
#                  WĘZEŁ POMOCNICZY: FAKE PLANNING NODE
# =====================================================================
add_executable(false_dv_path_planning_node
  false_dv_path_planning_node.cpp
  ${simSRC}
)

target_link_libraries(false_dv_path_planning_node PRIVATE
  ${catkin_LIBRARIES}
  nlohmann_json::nlohmann_json
  Eigen3::Eigen
  m
)

# =====================================================================
#                  WĘZEŁ POMOCNICZY: FAKE CONTROL NODE
# =====================================================================
add_executable(false_dv_control_node
  false_dv_control_node.cpp
)

target_link_libraries(false_dv_control_node PRIVATE
  ${catkin_LIBRARIES}
  m
)

# =====================================================================
#                         INSTALACJA BINAREK
# =====================================================================
install(TARGETS
  sim_loop_node
  false_dv_path_planning_node
  false_dv_control_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
