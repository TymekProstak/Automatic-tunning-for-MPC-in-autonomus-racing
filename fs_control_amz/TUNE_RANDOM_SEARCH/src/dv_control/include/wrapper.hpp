#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>

#include <ros/ros.h>
#include <ros/package.h>

#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <nlohmann/json.hpp>

#include "ParamBank.hpp"
#include "Vec2.hpp"
#include "dv_interfaces/Control.h"
#include "dv_interfaces/LtoResult.h"
#include "dv_interfaces/MPCDebug.h"
#include "dv_interfaces/RackAngleSensor.h"
#include "mpc_interface_amz_racing.hpp"
#include "spline.hpp"
#include "stanley.hpp"
#include "utilities.hpp"
#include "velocity_planner.hpp"

namespace v2_control {

class Controller {
public:
    Controller();
    Controller(ros::NodeHandle& nh, const ParamBank& param);

    void pathCallback(const dv_interfaces::LtoResult& msg);
    void odometryCallback(const nav_msgs::Odometry& msg);

private:
    // =====================================================
    // ROS
    // =====================================================
    ros::Subscriber path_sub;
    ros::Subscriber odom_sub;
    ros::Subscriber rack_angle_sub;

    ros::Publisher pub_control;
    ros::Publisher pub_geo_marker;
    ros::Publisher pub_ref_path;
    ros::Publisher pub_mpc_debug;
    ros::Publisher pub_gg_marker;

    // =====================================================
    // Internal state
    // =====================================================
    State current_state;
    MPCC_State mpc_state;
    ParamBank param_;

    Stanley stanley;
    MPCCInterface mpc;

    // =====================================================
    // Track / references
    // =====================================================
    VelocityPlannerResult ref_path;
    dv_interfaces::LtoResult lto_result;
    TrackSpline2D spline_complete_path;
    bool first_path_received = false;

    // =====================================================
    // Steering actuator model
    // =====================================================
    Eigen::Matrix2d Ad_maxon;
    Eigen::Vector2d Bd_maxon;

    // =====================================================
    // MPC memory / diagnostics
    // =====================================================
    double last_ddelta_opt_from_mpc = 0.0;
    double last_dthrothle_opt_from_mpc = 0.0;
    double last_mtv_opt_from_mpc = 0.0;

    double last_delta = 0.0;
    double last_s0 = 0.0;

    double ax_target_from_mpc = 0.0;
    double next_target_yaw_rate_from_mpc = 0.0;

    double ey_sum = 0.0;
    double epsi_sum = 0.0;
    double v_s_sum = 0.0;
    int control_loop_count = 0;

    double global_ey_ = 0.0;
    double global_epsi_ = 0.0;

    bool solver_failed_ = true;

    // =====================================================
    // Helpers
    // =====================================================
    void getCurrentState(const nav_msgs::Odometry& msg);
    void convert_state_to_mpc_state();

    void publishControlCommand(double steering_angle, double torque_request, double mtv);
    void publishLookaheadMarker(const geo_control_return& control_output);
    void publishReferencePath(const Eigen::VectorXd& X, const Eigen::VectorXd& Y);
    void publishGGLimitMarker();

    void geometricControl();
    void rackAngleCallback(const dv_interfaces::RackAngleSensor::ConstPtr& msg);
};

} // namespace v2_control