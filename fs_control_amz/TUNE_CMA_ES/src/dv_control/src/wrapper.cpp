#include "wrapper.hpp"
#include "ParamBank.hpp"

#include <chrono>
#include <limits>
#include <sstream>
#include <iomanip>

namespace v2_control
{

using json = nlohmann::json;

// ============================================================
//  CALLBACK TIMING
// ============================================================
namespace {
using Clock = std::chrono::steady_clock;

static inline double ms(const Clock::time_point& a, const Clock::time_point& b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Stat {
    double sum = 0.0;
    double mx  = 0.0;
    void add(double x) { sum += x; if (x > mx) mx = x; }
    void reset() { sum = 0.0; mx = 0.0; }
};

struct CallbackTiming {
    int n = 0;
    int window = 50;

    Stat total, getState, planner, convert, mpc;
    Stat dbgBuild, dbgPublish, pubCmd, pubRef, zoh, geom, gg;

    void reset_all()
    {
        n = 0;
        total.reset(); getState.reset(); planner.reset(); convert.reset();
        mpc.reset(); dbgBuild.reset(); dbgPublish.reset(); pubCmd.reset();
        pubRef.reset(); zoh.reset(); geom.reset(); gg.reset();
    }

    void print_and_reset()
    {
        const double inv = 1.0 / std::max(1, n);
        ROS_INFO_STREAM(
            "[CallbackTiming] window=" << n
            << " | total avg=" << total.sum*inv << " ms (max " << total.mx << ")"
            << " | getState avg=" << getState.sum*inv << " (max " << getState.mx << ")"
            << " | convert avg="  << convert.sum*inv  << " (max " << convert.mx << ")"
            << " | mpc avg="      << mpc.sum*inv      << " (max " << mpc.mx << ")"
            << " | dbgBuild avg=" << dbgBuild.sum*inv << " (max " << dbgBuild.mx << ")"
            << " | dbgPub avg="   << dbgPublish.sum*inv << " (max " << dbgPublish.mx << ")"
            << " | pubCmd avg="   << pubCmd.sum*inv   << " (max " << pubCmd.mx << ")"
            << " | pubRef avg="   << pubRef.sum*inv   << " (max " << pubRef.mx << ")"
            << " | geom avg="     << geom.sum*inv     << " (max " << geom.mx << ")"
            << " | GG avg="       << gg.sum*inv       << " (max " << gg.mx << ")"
        );
        reset_all();
    }
};

static CallbackTiming g_cb;

} // namespace

// =====================================================
//   CONSTRUCTORS
// =====================================================

Controller::Controller() {}

Controller::Controller(ros::NodeHandle &nh, const ParamBank &param)
    : stanley(param),
      mpc(param)
{
    current_state = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5};
    mpc_state = {};

    param_ = param;
    solver_failed_ = true;

    // -------------------------------------------------
    // Internal actuator / drive states held in wrapper
    // -------------------------------------------------
    mpc_state.delta         = 0.0;
    mpc_state.delta_dot     = 0.0;
    mpc_state.delta_request = 0.0;
    mpc_state.T             = 0.0;

    last_delta = 0.0;
    last_s0 = 0.0;
    global_epsi_ = 0.0;
    global_ey_   = 0.0;

    ax_target_from_mpc = 0.0;
    next_target_yaw_rate_from_mpc = 0.0;
    last_ddelta_opt_from_mpc = 0.0;
    last_dthrothle_opt_from_mpc = 0.0;
    last_mtv_opt_from_mpc = 0.0;

    const double omega = param_.get("model_steer_natural_freq");
    const double zeta  = param_.get("model_steer_damping");
    const double dt    = 1.0 / param_.get("odom_frequency");

    Eigen::Matrix2d A_cont;
    A_cont << 0.0, 1.0,
             -omega * omega, -2.0 * zeta * omega;

    Eigen::Vector2d B_cont(0.0, omega * omega);

    Ad_maxon = (A_cont * dt).exp();

    if (std::abs(A_cont.determinant()) > 1e-9) {
        Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        Bd_maxon = A_cont.inverse() * (Ad_maxon - I) * B_cont;
    } else {
        Bd_maxon = B_cont * dt;
    }

    path_sub = nh.subscribe("/path_planning/path", 1,
                            &Controller::pathCallback, this,
                            ros::TransportHints().tcpNoDelay());

    odom_sub = nh.subscribe("/ins/pose", 1,
                            &Controller::odometryCallback, this,
                            ros::TransportHints().tcpNoDelay());

    rack_angle_sub = nh.subscribe("/sensors/rack_angle", 1,
                                  &Controller::rackAngleCallback, this,
                                  ros::TransportHints().tcpNoDelay());

    pub_control     = nh.advertise<dv_interfaces::Control>("/dv_board/control", 1);
    pub_geo_marker  = nh.advertise<visualization_msgs::Marker>("/control/markers", 1);
    pub_ref_path    = nh.advertise<visualization_msgs::Marker>("/control/ref_path", 1);
    pub_mpc_debug   = nh.advertise<dv_interfaces::MPCDebug>("/control/mpc_debug", 1);
    pub_gg_marker   = nh.advertise<visualization_msgs::Marker>("/control/gg_limit_marker", 1);
}

// =====================================================
//   PATH CALLBACK
// =====================================================

void Controller::pathCallback(const dv_interfaces::LtoResult &msg)
{
    lto_result = msg;

    std::vector<double> xs, ys;
    xs.assign(lto_result.track_x.begin(), lto_result.track_x.end());
    ys.assign(lto_result.track_y.begin(), lto_result.track_y.end());

    std::vector<v2_control::Vec2> path_pts;
    path_pts.reserve(xs.size());
    ref_path.X_ref.resize(xs.size());
    ref_path.Y_ref.resize(xs.size());

    for (size_t i = 0; i < xs.size(); ++i) {
        path_pts.emplace_back((float)xs[i], (float)ys[i]);
        ref_path.X_ref(i) = xs[i];
        ref_path.Y_ref(i) = ys[i];
    }

    spline_complete_path.build(path_pts, true);
    mpc.setTrack(spline_complete_path);
    mpc.requestInitialGuessReset();

    first_path_received = true;
}

// =====================================================
//   ODOMETRY CALLBACK
// =====================================================

void Controller::odometryCallback(const nav_msgs::Odometry &msg)
{
    const auto t0 = Clock::now();

    const auto t_state0 = Clock::now();
    getCurrentState(msg);
    const auto t_state1 = Clock::now();

    const auto t_conv0 = Clock::now();
    convert_state_to_mpc_state();
    const auto t_conv1 = Clock::now();

    double ms_mpc     = 0.0;
    double ms_dbg_bld = 0.0;
    double ms_dbg_pub = 0.0;
    double ms_pub_cmd = 0.0;
    double ms_pub_ref = 0.0;
    double ms_geom    = 0.0;

    dv_interfaces::MPCDebug debug_msg;
    debug_msg.solver_failed = true;
    debug_msg.epsi_current = global_epsi_;
    debug_msg.ey_current   = global_ey_;

    const double kappa_here =
        spline_complete_path.valid() ? spline_complete_path.getCurvature(mpc_state.theta) : 0.0;

    debug_msg.kappa_ref = kappa_here;
    debug_msg.R_ref = (std::abs(kappa_here) > 1e-12) ? 1.0 / kappa_here : 0.0;

    debug_msg.v_s_current =
        current_state.vx * std::cos(global_epsi_) - current_state.vy * std::sin(global_epsi_);

    debug_msg.next_v_x_target = 0.0;
    debug_msg.next_vy_target = 0.0;
    debug_msg.ax_target = 0.0;
    debug_msg.ay_target = kappa_here * current_state.vx * current_state.vx;
    debug_msg.mpc_yaw_rate_from_curvature = current_state.vx * kappa_here;
    debug_msg.next_yaw_rate_target = 0.0;
    debug_msg.next_vtheta = 0.0f;
    debug_msg.next_vref   = 0.0f;

    auto publishMPCPathRed = [&](const Eigen::VectorXd& X, const Eigen::VectorXd& Y)
    {
        if (X.size() < 2 || Y.size() < 2 || X.size() != Y.size()) return;

        visualization_msgs::Marker line;
        line.header.frame_id = "map";
        line.header.stamp    = ros::Time::now();
        line.ns = "mpc_path_line";
        line.id = 999999;
        line.type   = visualization_msgs::Marker::LINE_STRIP;
        line.action = visualization_msgs::Marker::ADD;
        line.pose.orientation.w = 1.0;
        line.scale.x = 0.06;
        line.color.r = 1.0;
        line.color.g = 0.0;
        line.color.b = 0.0;
        line.color.a = 1.0;
        line.lifetime = ros::Duration(0);
        line.frame_locked = true;
        line.points.reserve((size_t)X.size());

        for (int i = 0; i < X.size(); ++i) {
            if (!std::isfinite(X(i)) || !std::isfinite(Y(i))) continue;
            geometry_msgs::Point p;
            p.x = X(i);
            p.y = Y(i);
            p.z = 0.08;
            line.points.push_back(p);
        }
        pub_ref_path.publish(line);
    };

    auto deleteMPCPathRed = [&]()
    {
        visualization_msgs::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = ros::Time::now();
        del.ns = "mpc_path_line";
        del.id = 999999;
        del.action = visualization_msgs::Marker::DELETE;
        pub_ref_path.publish(del);
    };

    auto deleteMPCSampledOrange = [&]()
    {
        visualization_msgs::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = ros::Time::now();
        del.ns = "mpc_sampled_path_debug";
        del.id = 999998;
        del.action = visualization_msgs::Marker::DELETE;
        pub_ref_path.publish(del);
    };

    const bool can_run_mpc = first_path_received && spline_complete_path.valid();

    if (can_run_mpc)
    {
        MPCC_Return mpc_return;
        {
            const auto t = Clock::now();
            mpc_return = mpc.solve(mpc_state, mpc_state.theta);
            ms_mpc += ms(t, Clock::now());
        }

        if (mpc_return.success)
        {
            solver_failed_ = false;
            debug_msg.solver_failed = false;

            debug_msg.next_yaw_rate_target = mpc_return.next_yaw_rate;
            debug_msg.next_v_x_target      = mpc_return.next_vx_target;
            debug_msg.next_vy_target       = mpc_return.next_vy_target;
            debug_msg.ax_target            = mpc_return.ax;
            debug_msg.ay_target            = mpc_return.ay;
            debug_msg.next_vtheta          = static_cast<float>(mpc_return.next_vtheta);
            debug_msg.next_vref            = static_cast<float>(mpc_return.next_vref);

            last_ddelta_opt_from_mpc      = mpc_return.ddelta_request;
            last_dthrothle_opt_from_mpc   = mpc_return.dT;
            last_mtv_opt_from_mpc         = mpc_return.Mtv;
            next_target_yaw_rate_from_mpc = mpc_return.next_yaw_rate;
            ax_target_from_mpc            = mpc_return.ax;

            {
                const auto t = Clock::now();

                const double delta_request_to_publish = std::clamp(
                    mpc_return.next_delta_request,
                    param_.get("min_delta"),
                    param_.get("max_delta")
                );

                const double T_next = std::clamp(
                    mpc_return.next_T,
                    param_.get("mpc_bounds_min_T"),
                    param_.get("mpc_bounds_max_T")
                );

                const double torque_request = 100.0 * T_next;

                publishControlCommand(delta_request_to_publish, torque_request, last_mtv_opt_from_mpc);

                // -------------------------------------------------
                // IMPORTANT:
                // wrapper keeps internal predicted steering states
                // from MPC, NOT from sensor
                // -------------------------------------------------
                mpc_state.delta         = mpc_return.next_delta;
                mpc_state.delta_dot     = mpc_return.next_delta_dot;
                mpc_state.delta_request = delta_request_to_publish;
                mpc_state.T             = T_next;

                last_delta = delta_request_to_publish;

                ms_pub_cmd += ms(t, Clock::now());
            }

            {
                const auto t = Clock::now();
                publishReferencePath(ref_path.X_ref, ref_path.Y_ref);
                publishMPCPathRed(mpc_return.X_mpc, mpc_return.Y_mpc);
                ms_pub_ref += ms(t, Clock::now());
            }
        }
        else
        {
            solver_failed_ = true;
            debug_msg.solver_failed = true;
            debug_msg.next_yaw_rate_target = 0.0;
            debug_msg.next_v_x_target = 0.0;
            debug_msg.next_vy_target = 0.0;
            debug_msg.ax_target = 0.0;
            debug_msg.ay_target = 0.0;
            debug_msg.next_vtheta = 0.0f;
            debug_msg.next_vref   = 0.0f;

            next_target_yaw_rate_from_mpc = 0.0;
            ax_target_from_mpc = 0.0;
            last_ddelta_opt_from_mpc = 0.0;
            last_dthrothle_opt_from_mpc = 0.0;
            last_mtv_opt_from_mpc = 0.0;

            deleteMPCPathRed();
            deleteMPCSampledOrange();

            {
                const auto t = Clock::now();
                stanley.setTrack(ref_path.X_ref, ref_path.Y_ref);
                convert_state_to_mpc_state();
                geometricControl();
                ms_geom += ms(t, Clock::now());
            }
        }

        {
            const auto t = Clock::now();
            std::vector<double> scx, scy;
            mpc.getLastSampledPath(scx, scy);

            if (!scx.empty() && scx.size() == scy.size()) {
                visualization_msgs::Marker line;
                line.header.frame_id = "map";
                line.header.stamp    = ros::Time::now();
                line.ns = "mpc_sampled_path_debug";
                line.id = 999998;
                line.type   = visualization_msgs::Marker::LINE_STRIP;
                line.action = visualization_msgs::Marker::ADD;
                line.pose.orientation.w = 1.0;
                line.scale.x = 0.06;
                line.color.r = 1.0;
                line.color.g = 0.55;
                line.color.b = 0.0;
                line.color.a = 1.0;
                line.lifetime = ros::Duration(0);
                line.frame_locked = true;
                line.points.reserve(scx.size());

                for (size_t i = 0; i < scx.size(); ++i) {
                    if (!std::isfinite(scx[i]) || !std::isfinite(scy[i])) continue;
                    geometry_msgs::Point p;
                    p.x = scx[i];
                    p.y = scy[i];
                    p.z = 0.06;
                    line.points.push_back(p);
                }
                pub_ref_path.publish(line);
            }
            ms_pub_ref += ms(t, Clock::now());
        }

        ey_sum   += global_ey_   * global_ey_;
        epsi_sum += global_epsi_ * global_epsi_;
        v_s_sum  += current_state.vx * std::cos(global_epsi_)
                  - current_state.vy * std::sin(global_epsi_);
        control_loop_count++;

        debug_msg.ey_avg   = std::sqrt(ey_sum   / control_loop_count);
        debug_msg.epsi_avg = std::sqrt(epsi_sum / control_loop_count);
        debug_msg.v_s_avg  = v_s_sum / control_loop_count;
    }
    else
    {
        solver_failed_ = true;

        debug_msg.solver_failed = true;
        debug_msg.next_yaw_rate_target = 0.0;
        debug_msg.next_v_x_target = 0.0;
        debug_msg.next_vy_target = 0.0;
        debug_msg.ax_target = 0.0;
        debug_msg.ay_target = 0.0;
        debug_msg.next_vtheta = 0.0f;
        debug_msg.next_vref   = 0.0f;

        next_target_yaw_rate_from_mpc = 0.0;
        ax_target_from_mpc = 0.0;
        last_ddelta_opt_from_mpc = 0.0;
        last_dthrothle_opt_from_mpc = 0.0;
        last_mtv_opt_from_mpc = 0.0;

        deleteMPCPathRed();
        deleteMPCSampledOrange();

        const auto t = Clock::now();
        if (ref_path.X_ref.size() > 1) {
            stanley.setTrack(ref_path.X_ref, ref_path.Y_ref);
            convert_state_to_mpc_state();
            geometricControl();
        }
        ms_geom += ms(t, Clock::now());
    }

    const auto t_gg0 = Clock::now();
    publishGGLimitMarker();
    const auto t_gg1 = Clock::now();

    {
        const auto t = Clock::now();
        pub_mpc_debug.publish(debug_msg);
        ms_dbg_pub += ms(t, Clock::now());
    }

    const auto t1 = Clock::now();

    g_cb.n++;
    g_cb.total.add(ms(t0, t1));
    g_cb.getState.add(ms(t_state0, t_state1));
    g_cb.convert.add(ms(t_conv0, t_conv1));
    g_cb.mpc.add(ms_mpc);
    g_cb.dbgBuild.add(ms_dbg_bld);
    g_cb.dbgPublish.add(ms_dbg_pub);
    g_cb.pubCmd.add(ms_pub_cmd);
    g_cb.pubRef.add(ms_pub_ref);
    g_cb.geom.add(ms_geom);
    g_cb.gg.add(ms(t_gg0, t_gg1));

    if (g_cb.n >= g_cb.window) {
        g_cb.print_and_reset();
    }
}

// =====================================================
//   GEOMETRIC CONTROL
// =====================================================

void Controller::geometricControl()
{
    geo_control_return control_output;

    if (static_cast<int>(param_.get("using_stanley")))
    {
        control_output = stanley.StanleyControl(current_state);
    }

    control_output.steering_angle =
        std::clamp(control_output.steering_angle,
                   param_.get("min_delta"),
                   param_.get("max_delta"));

    publishControlCommand(control_output.steering_angle, 0.0, 0.0);
    publishLookaheadMarker(control_output);
    publishReferencePath(ref_path.X_ref, ref_path.Y_ref);

    // -------------------------------------------------
    // On geometric fallback:
    // set internal actuator state consistently
    // -------------------------------------------------
    mpc_state.delta         = control_output.steering_angle;
    mpc_state.delta_dot     = 0.0;
    mpc_state.delta_request = control_output.steering_angle;

    last_delta = control_output.steering_angle;
}

// =====================================================
//   PUBLISH CONTROL
// =====================================================

void Controller::publishControlCommand(double steering_angle, double torque_request, double mtv)
{
    dv_interfaces::Control controlMsg;

    if (solver_failed_)
    {
        controlMsg.movement = static_cast<float>(param_.get("v_target"));
        controlMsg.steeringAngle_rad = static_cast<float>(
            std::clamp(steering_angle, param_.get("min_delta"), param_.get("max_delta"))
        );
        controlMsg.finished = false;
        controlMsg.move_type = dv_interfaces::Control::SPEED_KMH;
        controlMsg.serviceBrake = 0;
        controlMsg.current_speed = 0;
        controlMsg.fx_target = 0.0;
        controlMsg.mz_target = 0.0;
        controlMsg.ax_target = 0.0;
        controlMsg.next_yaw_rate_target = 0.0;
        pub_control.publish(controlMsg);
    }
    else
    {
        controlMsg.movement = static_cast<float>(torque_request);
        controlMsg.steeringAngle_rad = static_cast<float>(
            std::clamp(steering_angle, param_.get("min_delta"), param_.get("max_delta"))
        );
        controlMsg.finished = false;
        controlMsg.move_type = dv_interfaces::Control::TORQUE_PERCENTAGE;
        controlMsg.serviceBrake = 0;
        controlMsg.fx_target = torque_request / 100.0 * 4.0
                             * param_.get("model_max_motor_torque")
                             / param_.get("model_wheel_radius");
        controlMsg.mz_target = mtv;
        controlMsg.ax_target = static_cast<float>(ax_target_from_mpc);
        controlMsg.next_yaw_rate_target = static_cast<float>(next_target_yaw_rate_from_mpc);
        controlMsg.current_speed = static_cast<float>(current_state.vx);
        pub_control.publish(controlMsg);
    }
}

// =====================================================
//   VEHICLE STATE
// =====================================================

void Controller::getCurrentState(const nav_msgs::Odometry &msg)
{
    current_state.X = msg.pose.pose.position.x;
    current_state.Y = msg.pose.pose.position.y;

    tf2::Quaternion q(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w
    );

    double roll, pitch, phi;
    tf2::Matrix3x3(q).getRPY(roll, pitch, phi);

    double yaw = phi;
    unwrap_angle(yaw);
    current_state.yaw = yaw;

    const double vx_msg = msg.twist.twist.linear.x;
    const double vy_msg = msg.twist.twist.linear.y;

    current_state.vx =  vx_msg * std::cos(yaw) + vy_msg * std::sin(yaw);
    current_state.vy = -vx_msg * std::sin(yaw) + vy_msg * std::cos(yaw);

    current_state.yaw_rate = msg.twist.twist.angular.z;
}

// =====================================================
//   STATE CONVERSION
//   Global/cartesian -> wrapper-facing MPCC_State
//   Steering internal states are NOT overwritten here.
// =====================================================

static inline double wrap_to_pi(double a)
{
    while (a >  M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

void Controller::convert_state_to_mpc_state()
{
    if (!spline_complete_path.valid()) {
        ROS_WARN_STREAM_THROTTLE(2.0, "[MPC] Spline not valid");
        return;
    }

    v2_control::Vec2 q(current_state.X, current_state.Y);
    double s_proj = spline_complete_path.projectToSpline(q);
    last_s0 = s_proj;

    const double path_yaw = spline_complete_path.getYaw(s_proj);
    const double epsi = wrap_to_pi(current_state.yaw - path_yaw);
    const double ey   = spline_complete_path.signedNormalDistance(q, s_proj);

    global_epsi_ = epsi;
    global_ey_   = ey;

    mpc_state.X     = current_state.X;
    mpc_state.Y     = current_state.Y;
    mpc_state.phi   = current_state.yaw;
    mpc_state.vx    = current_state.vx;
    mpc_state.vy    = current_state.vy;
    mpc_state.r     = current_state.yaw_rate;
    mpc_state.theta = s_proj;

    // NOTE:
    // mpc_state.delta / delta_dot / delta_request / T
    // stay as internally propagated wrapper states
}

// =====================================================
//   MARKERS
// =====================================================

void Controller::publishLookaheadMarker(const geo_control_return &control_output)
{
    visualization_msgs::Marker lookahead_marker;
    lookahead_marker.header.frame_id = "map";
    lookahead_marker.header.stamp    = ros::Time(0);
    lookahead_marker.ns  = "lookahead_point";
    lookahead_marker.id  = 1;
    lookahead_marker.type = visualization_msgs::Marker::CUBE;
    lookahead_marker.action = visualization_msgs::Marker::ADD;
    lookahead_marker.pose.position.x = control_output.look_ahead_point.x;
    lookahead_marker.pose.position.y = control_output.look_ahead_point.y;
    lookahead_marker.pose.position.z = 0.05;
    lookahead_marker.pose.orientation.w = 1.0;
    lookahead_marker.scale.x = 0.2;
    lookahead_marker.scale.y = 0.2;
    lookahead_marker.scale.z = 0.01;
    lookahead_marker.color.a = 1.0;
    lookahead_marker.color.r = 1.0;
    pub_geo_marker.publish(lookahead_marker);
}

void Controller::publishReferencePath(const Eigen::VectorXd &X,
                                      const Eigen::VectorXd &Y)
{
    for (int i = 0; i < X.size(); ++i)
    {
        visualization_msgs::Marker m;
        m.header.frame_id = "map";
        m.header.stamp    = ros::Time::now();
        m.ns = "ref_path_points";
        m.id = i + 200000;
        m.type   = visualization_msgs::Marker::SPHERE;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.position.x = X(i);
        m.pose.position.y = Y(i);
        m.pose.position.z = 0.05;
        m.scale.x = 0.10;
        m.scale.y = 0.10;
        m.scale.z = 0.10;
        m.color.r = 0.0;
        m.color.g = 1.0;
        m.color.b = 0.0;
        m.color.a = 1.0;
        m.lifetime = ros::Duration(0);
        m.frame_locked = true;
        pub_ref_path.publish(m);
    }
}

// =====================================================
//   RACK ANGLE CALLBACK
//   For now intentionally ignored.
//   Internal actuator state is propagated by MPC/wrapper.
// =====================================================

void Controller::rackAngleCallback(const dv_interfaces::RackAngleSensor::ConstPtr& msg)
{
    (void)msg;
}

void Controller::publishGGLimitMarker()
{
    visualization_msgs::Marker msg;
    msg.header.frame_id = "gg_dashboard";
    msg.header.stamp    = ros::Time::now();
    msg.ns = "gg_envelope";
    msg.id = 0;
    msg.type   = visualization_msgs::Marker::LINE_STRIP;
    msg.action = visualization_msgs::Marker::ADD;
    msg.scale.x = 1.03;
    msg.color.r = 0.5;
    msg.color.g = 0.5;
    msg.color.b = 0.5;
    msg.color.a = 1.0;

    const double max_ax = param_.get("model_mux") * 9.81;
    const double max_ay = param_.get("model_muy") * 9.81;

    for (int i = 0; i <= 100; ++i) {
        const double th = (i / 100.0) * 2.0 * M_PI;
        geometry_msgs::Point p;
        p.x = max_ay * std::cos(th);
        p.y = max_ax * std::sin(th);
        p.z = 0.0;
        msg.points.push_back(p);
    }

    pub_gg_marker.publish(msg);
}

} // namespace v2_control