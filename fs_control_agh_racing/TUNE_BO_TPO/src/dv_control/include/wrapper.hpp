#pragma once 

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "stanley.hpp"
#include "dv_interfaces/Control.h"
#include "ParamBank.hpp"
#include "utilities.hpp"
#include "mpc_interface_agh_racing.hpp"
#include "dv_interfaces/MPCDebug.h"
#include "dv_interfaces/LtoResult.h"
#include "spline.hpp"
#include "Vec2.hpp"
#include "velocity_planner.hpp"

#include <unsupported/Eigen/MatrixFunctions> 

namespace v2_control {

using json = nlohmann::json;

class Controller {
public:
    Controller();
    Controller(ros::NodeHandle &nh, const ParamBank &param);

    void pathCallback(const dv_interfaces::LtoResult& msg);
    void odometryCallback(const nav_msgs::Odometry& msg);

private:
    ros::Subscriber path_sub;
    ros::Subscriber odom_sub;

    ros::Publisher pub_control;
    ros::Publisher pub_geo_marker;
    ros::Publisher pub_ref_path;
    ros::Publisher pub_mpc_debug;
    ros::Publisher pub_gg_marker;

    State current_state;
    MPC_State mpc_state;
    ParamBank param_;

    void getCurrentState (const nav_msgs::Odometry& msg);
    void publishControlCommand (double steering_angle, double torque_request, double mtv);
    void publishLookaheadMarker (const geo_control_return &control_output);
    void publishReferencePath(const Eigen::VectorXd &X, const Eigen::VectorXd &Y);
    void geometricControl();

    void convert_state_to_mpc_state();
    void publishGGLimitMarker();

    void ZOH_for_steering();
    double acc_to_throttle_percentage(double a_desired);

    Eigen::Matrix2d Ad_maxon;
    Eigen::Vector2d Bd_maxon;

    Stanley stanley;

    Eigen::VectorXd X_last_from_pp;
    Eigen::VectorXd Y_last_from_pp;

    double last_ddelta_opt_from_mpc = 0.0;
    double last_dthrothle_opt_from_mpc = 0.0;
    double last_delta = 0.0;
    double last_throttle = 0.0;

    double next_vx_target_from_mpc = 0.0;
    double ay_target_from_mpc = 0.0;
    double ax_target_from_mpc = 0.0;
    double next_yaw_rate_target_from_mpc = 0.0;

    MPCInterface mpc;
    VelocityPlannerResult ref_path;

    double ey_sum = 0.0;
    double epsi_sum = 0.0;
    double v_s_sum = 0.0;
    int control_loop_count = 0;

    bool solver_failed_ = false;
    bool mpc_eligible_ = false;

    double last_mtv_opt_from_mpc = 0.0;
    double next_target_yaw_rate_from_mpc = 0.0;

    double last_s0 = 0.0;

    dv_interfaces::LtoResult lto_result;
    TrackSpline2D spline_complete_path;
};

} // namespace v2_control