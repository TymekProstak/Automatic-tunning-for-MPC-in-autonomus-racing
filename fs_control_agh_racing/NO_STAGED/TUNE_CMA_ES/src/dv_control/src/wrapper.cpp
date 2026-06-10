#include "wrapper.hpp"
#include <chrono>
#include <limits>
#include <sstream>
#include <iomanip>

namespace v2_control
{

using json = nlohmann::json;

// ============================================================
//  CALLBACK TIMING (windowed avg + max, bez spamu co callback)
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
    int window = 50; // <- ile callbacków do uśrednienia

    Stat total;
    Stat getState;
    Stat planner;
    Stat convert;
    Stat mpc;
    Stat dbgBuild;
    Stat dbgPublish;
    Stat pubCmd;
    Stat pubRef;
    Stat zoh;
    Stat geom;
    Stat gg;

    void reset_all() {
        n = 0;
        total.reset();
        getState.reset();
        planner.reset();
        convert.reset();
        mpc.reset();
        dbgBuild.reset();
        dbgPublish.reset();
        pubCmd.reset();
        pubRef.reset();
        zoh.reset();
        geom.reset();
        gg.reset();
    }

    void print_and_reset()
    {
        const double inv = 1.0 / std::max(1, n);

        ROS_INFO_STREAM(
            "[CallbackTiming] window=" << n
            << " | total avg=" << total.sum*inv << " ms (max " << total.mx << ")"
            << " | getState avg=" << getState.sum*inv << " (max " << getState.mx << ")"
            << " | planner avg="  << planner.sum*inv  << " (max " << planner.mx << ")"
            << " | convert avg="  << convert.sum*inv  << " (max " << convert.mx << ")"
            << " | mpc avg="      << mpc.sum*inv      << " (max " << mpc.mx << ")"
            << " | dbgBuild avg=" << dbgBuild.sum*inv << " (max " << dbgBuild.mx << ")"
            << " | dbgPub avg="   << dbgPublish.sum*inv << " (max " << dbgPublish.mx << ")"
            << " | pubCmd avg="   << pubCmd.sum*inv   << " (max " << pubCmd.mx << ")"
            << " | pubRef avg="   << pubRef.sum*inv   << " (max " << pubRef.mx << ")"
            << " | ZOH avg="      << zoh.sum*inv      << " (max " << zoh.mx << ")"
            << " | geom avg="     << geom.sum*inv     << " (max " << geom.mx << ")"
            << " | GG avg="       << gg.sum*inv       << " (max " << gg.mx << ")"
        );

        reset_all();
    }
};

static CallbackTiming g_cb;

// ============================================================
//  DEBUG: range/min/max + NaN/Inf for Eigen vectors
// ============================================================
struct VecStats {
    double mn = 0.0;
    double mx = 0.0;
    bool any_nan = false;
    bool any_inf = false;
    int n = 0;
};

static inline VecStats stats_vec(const Eigen::VectorXd& v)
{
    VecStats s;
    s.n = (int)v.size();
    if (s.n == 0) return s;

    s.mn =  std::numeric_limits<double>::infinity();
    s.mx = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < s.n; ++i) {
        const double x = v(i);
        if (std::isnan(x)) s.any_nan = true;
        if (!std::isfinite(x)) s.any_inf = true;

        if (std::isfinite(x)) {
            if (x < s.mn) s.mn = x;
            if (x > s.mx) s.mx = x;
        }
    }

    if (!std::isfinite(s.mn)) s.mn = 0.0;
    if (!std::isfinite(s.mx)) s.mx = 0.0;

    return s;
}

static inline void print_ref_ranges_throttled(
    const Eigen::VectorXd& v_ref,
    const Eigen::VectorXd& a_ref,
    const Eigen::VectorXd& kappa_ref,
    double throttle_sec = 0.5)
{
    static ros::Time last = ros::Time(0);
    const ros::Time now = ros::Time::now();
    if ((now - last).toSec() < throttle_sec) return;
    last = now;

    const auto sv = stats_vec(v_ref);
    const auto sa = stats_vec(a_ref);
    const auto sk = stats_vec(kappa_ref);

    auto head3 = [](const Eigen::VectorXd& v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << "[";
        const int n = std::min<int>(3, (int)v.size());
        for (int i = 0; i < n; ++i) {
            oss << v(i);
            if (i+1 < n) oss << ", ";
        }
        oss << "]";
        return oss.str();
    };

    ROS_WARN_STREAM(std::fixed << std::setprecision(4)
        << "[MPC REF] "
        << "v_ref: n=" << sv.n << " min=" << sv.mn << " max=" << sv.mx
        << " nan=" << sv.any_nan << " inf=" << sv.any_inf
        << " head=" << head3(v_ref)
        << " | a_ref: n=" << sa.n << " min=" << sa.mn << " max=" << sa.mx
        << " nan=" << sa.any_nan << " inf=" << sa.any_inf
        << " head=" << head3(a_ref)
        << " | kappa: n=" << sk.n << " min=" << sk.mn << " max=" << sk.mx
        << " nan=" << sk.any_nan << " inf=" << sk.any_inf
        << " head=" << head3(kappa_ref)
    );
}

static inline bool finite7_mpc_state(const MPC_State& s)
{
    return std::isfinite(s.ey) && std::isfinite(s.epsi) && std::isfinite(s.vy) &&
           std::isfinite(s.r) && std::isfinite(s.delta) && std::isfinite(s.d_delta) &&
           std::isfinite(s.delta_request);
}

} // namespace

// =======================
//   K O N S T R U K T O R Y
// =======================
Controller::Controller()
{
    std::cout << "default constructor, not everything is initialized properly" << std::endl;
}

Controller::Controller(ros::NodeHandle &nh, const ParamBank &param):
    stanley(param),
    mpc(param)
{
    // --------------------------
    // WSTĘPNY STAN POJAZDU
    // --------------------------
    current_state = {
        0.0, // X
        0.0, // Y
        0.0, // yaw
        0.0, // delta
        0.0, // delta_dot
        0.0, // vx
        0.0, // vy
        0.0, // yaw_rate
        0.5  // acc
    };

    mpc_state = {
        0.0, // ey
        0.0, // epsi
        0.0, // vy
        0.0, // r
        0.0, // delta
        0.0, // d_delta
        0.0  // delta_request
    };

    param_ = param;

    // ==========================================
    // PRE-KOMPUTACJA DYNAMIKI AKTUATORA (ZOH)
    // ==========================================
    double omega = param_.get("model_steer_natural_freq");
    double zeta  = param_.get("model_steer_damping");
    double dt    = 1.0 / param_.get("odom_frequency");

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

    // ===== SUBY =====
    path_sub = nh.subscribe("/path_planning/path", 1,
                            &Controller::pathCallback, this,
                            ros::TransportHints().tcpNoDelay());

    odom_sub = nh.subscribe("/ins/pose", 1,
                            &Controller::odometryCallback, this,
                            ros::TransportHints().tcpNoDelay());

    // ===== PUBY =====
    pub_control     = nh.advertise<dv_interfaces::Control>("/dv_board/control", 1);
    pub_geo_marker  = nh.advertise<visualization_msgs::Marker>("/control/markers", 1);
    pub_ref_path    = nh.advertise<visualization_msgs::Marker>("/control/ref_path", 1);
    pub_mpc_debug   = nh.advertise<dv_interfaces::MPCDebug>("/control/mpc_debug", 1);
    pub_gg_marker   = nh.advertise<visualization_msgs::Marker>("/control/gg_limit_marker", 1);
}

// =====================================================
//   CALLBACK PATH
// =====================================================
void Controller::pathCallback(const dv_interfaces::LtoResult &msg)
{
    const std::size_t Nx = msg.track_x.size();
    const std::size_t Ny = msg.track_y.size();
    const std::size_t Np = std::min(Nx, Ny);

    if (Np < 3) {
        return;
    }

    std::vector<Vec2> path_from_lto;
    path_from_lto.reserve(Np);
    for (std::size_t i = 0; i < Np; ++i) {
        path_from_lto.emplace_back(
            static_cast<float>(msg.track_x[i]),
            static_cast<float>(msg.track_y[i])
        );
    }

    X_last_from_pp = Eigen::Map<const Eigen::VectorXd>(msg.track_x.data(), static_cast<int>(Np));
    Y_last_from_pp = Eigen::Map<const Eigen::VectorXd>(msg.track_y.data(), static_cast<int>(Np));

    stanley.setTrack(X_last_from_pp, Y_last_from_pp);

    spline_complete_path.build(path_from_lto, /*closed_loop=*/true);
    mpc.set_Spline(spline_complete_path);
}

// =====================================================
//   CALLBACK ODOMETRII (z timingiem, windowed)
// =====================================================
void Controller::odometryCallback(const nav_msgs::Odometry &msg)
{
    const auto t0 = Clock::now();

    // ---------------------------
    // 1) getCurrentState
    // ---------------------------
    const auto t_state0 = Clock::now();
    getCurrentState(msg);
    const auto t_state1 = Clock::now();

    // ---------------------------
    // 1b) convert_state_to_mpc_state
    // ---------------------------
    const auto t_conv0 = Clock::now();
    convert_state_to_mpc_state();
    const auto t_conv1 = Clock::now();

    // ---------------------------
    // 2) velocity planner
    // ---------------------------
    const auto t_plan0 = Clock::now();
    if (spline_complete_path.valid() && spline_complete_path.isClosed()) {
        ref_path = velocity_planner_process_for_control(param_, spline_complete_path, current_state, last_s0);
    }
    const auto t_plan1 = Clock::now();

    double ms_mpc     = 0.0;
    double ms_dbg_bld = 0.0;
    double ms_dbg_pub = 0.0;
    double ms_pub_cmd = 0.0;
    double ms_pub_ref = 0.0;
    double ms_zoh     = 0.0;
    double ms_geom    = 0.0;

    if (ref_path.valid)
    {
        if (current_state.vx > 2.0)
        {
            // ============================================================
            // MPC solve
            // ============================================================
            MPC_Return mpc_return;
            {
                const auto t = Clock::now();

                // // DEBUG: zakresy ref (co 0.5 s)
                // print_ref_ranges_throttled(ref_path.velocity_ref,
                //                            ref_path.acceleration_ref,
                //                            ref_path.curvature,
                //                            0.5);

                // // DEBUG: sanity-check stanu wejściowego
                // if (!finite7_mpc_state(mpc_state)) {
                //     ROS_ERROR_STREAM("[MPC] mpc_state has NaN/INF: ey=" << mpc_state.ey
                //         << " epsi=" << mpc_state.epsi
                //         << " vy=" << mpc_state.vy
                //         << " r=" << mpc_state.r
                //         << " delta=" << mpc_state.delta
                //         << " d_delta=" << mpc_state.d_delta
                //         << " d_req=" << mpc_state.delta_request);
                // }

                // UWAGA: NIE ROBIĆ "MPC_Return mpc_return = ..." (shadowing)
               mpc_return = mpc.solve(mpc_state, spline_complete_path, last_s0, ref_path.velocity_ref, ref_path.acceleration_ref, current_state.vx);
                ms_mpc += ms(t, Clock::now());
            }

            // ============================================================
            // debug msg build
            // ============================================================
            dv_interfaces::MPCDebug debug_msg;
            {
                const auto t = Clock::now();

                debug_msg.epsi_current = mpc_state.epsi;
                debug_msg.ey_current   = mpc_state.ey;

                debug_msg.v_s_current  = current_state.vx * std::cos(mpc_state.epsi)
                                       - current_state.vy * std::sin(mpc_state.epsi);

                debug_msg.kappa_ref = ref_path.curvature(0);
                debug_msg.R_ref = (std::abs(ref_path.curvature(0)) > 1e-6) ? (1.0 / std::abs(ref_path.curvature(0))) : 0.0;

                debug_msg.next_v_x_target = ref_path.velocity_ref(1);
                debug_msg.ax_target       = ref_path.acceleration_ref(1);

                debug_msg.ay_target       = current_state.vx * current_state.vx * ref_path.curvature(1);
                debug_msg.mpc_yaw_rate_from_curvature = current_state.vx * ref_path.curvature(1);

                ms_dbg_bld += ms(t, Clock::now());
            }

            if (mpc_return.success)
            {
                debug_msg.solver_failed = false;
                debug_msg.next_yaw_rate_target = mpc_return.next_yaw_rate;

                last_ddelta_opt_from_mpc = mpc_return.ddelta_opt;
                last_mtv_opt_from_mpc    = mpc_return.mtv_opt;
                next_target_yaw_rate_from_mpc = mpc_return.next_yaw_rate;

                // publishControlCommand
                {
                    const auto t = Clock::now();

                    const double d_delta_request = last_ddelta_opt_from_mpc / param_.get("odom_frequency");
                    double delta_request = mpc_state.delta_request + d_delta_request;
                    delta_request = std::clamp(delta_request, param_.get("min_delta"), param_.get("max_delta"));
                    mpc_state.delta_request = delta_request;

                    const double aceleration_request = ref_path.acceleration_ref(1);
                    const double torque_request = acc_to_throttle_percentage(aceleration_request);

                    publishControlCommand(delta_request, torque_request, last_mtv_opt_from_mpc);

                    ms_pub_cmd += ms(t, Clock::now());
                }

                // publishReferencePath
                {
                    const auto t = Clock::now();
                    publishReferencePath(ref_path.X_ref, ref_path.Y_ref);
                    ms_pub_ref += ms(t, Clock::now());
                }

                // ZOH
                {
                    const auto t = Clock::now();
                    ZOH_for_steering();
                    ms_zoh += ms(t, Clock::now());
                }
            }
            else
            {
                debug_msg.solver_failed = true;
                debug_msg.next_yaw_rate_target = 0.0;

                // geometria po failu MPC
                {
                    const auto t = Clock::now();

                    stanley.setTrack(ref_path.X_ref, ref_path.Y_ref);
                    convert_state_to_mpc_state();
                    geometricControl();
                    ZOH_for_steering();

                    ms_geom += ms(t, Clock::now());
                }
            }

            // RMSE-y do debug
            ey_sum   += mpc_state.ey   * mpc_state.ey;
            epsi_sum += mpc_state.epsi * mpc_state.epsi;
            v_s_sum  += debug_msg.v_s_current;
            control_loop_count++;

            debug_msg.ey_avg   = std::sqrt(ey_sum   / control_loop_count);
            debug_msg.epsi_avg = std::sqrt(epsi_sum / control_loop_count);
            debug_msg.v_s_avg  = v_s_sum / control_loop_count;

            // publish debug
            {
                const auto t = Clock::now();
                pub_mpc_debug.publish(debug_msg);
                ms_dbg_pub += ms(t, Clock::now());
            }
        }
        else
        {
            // geometria (vx <= 2)
            const auto t = Clock::now();

            stanley.setTrack(ref_path.X_ref, ref_path.Y_ref);
            convert_state_to_mpc_state();
            geometricControl();
            ZOH_for_steering();

            ms_geom += ms(t, Clock::now());
        }
    }

    // GG marker
    const auto t_gg0 = Clock::now();
    publishGGLimitMarker();
    const auto t_gg1 = Clock::now();

    const auto t1 = Clock::now();

    // ===========================
    // accumulate + print co window
    // ===========================
    g_cb.n++;
    g_cb.total.add(ms(t0, t1));
    g_cb.getState.add(ms(t_state0, t_state1));
    g_cb.convert.add(ms(t_conv0, t_conv1));
    g_cb.planner.add(ms(t_plan0, t_plan1));
    g_cb.mpc.add(ms_mpc);
    g_cb.dbgBuild.add(ms_dbg_bld);
    g_cb.dbgPublish.add(ms_dbg_pub);
    g_cb.pubCmd.add(ms_pub_cmd);
    g_cb.pubRef.add(ms_pub_ref);
    g_cb.zoh.add(ms_zoh);
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
        std::clamp(control_output.steering_angle, param_.get("min_delta"), param_.get("max_delta"));

    double aceleration_request = ref_path.acceleration_ref(1);
    const double torque_request = acc_to_throttle_percentage(aceleration_request);

    publishControlCommand(control_output.steering_angle, torque_request, 0.0);
    publishLookaheadMarker(control_output);
    publishReferencePath(ref_path.X_ref, ref_path.Y_ref);

    mpc_state.delta_request = control_output.steering_angle;
    ZOH_for_steering();
}

// =====================================================
//   PUBLISH CONTROL
// =====================================================
void Controller::publishControlCommand(double steering_angle, double torque_request, double mtv)
{
    dv_interfaces::Control controlMsg;

    if (current_state.vx < 2.0)
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

        controlMsg.fx_target = torque_request/100*4*param_.get("model_max_motor_torque")/param_.get("model_wheel_radius");
        controlMsg.mz_target = mtv;

        controlMsg.ax_target = static_cast<float>(ref_path.acceleration_ref(1));
        controlMsg.next_yaw_rate_target = static_cast<float>(next_target_yaw_rate_from_mpc);
        controlMsg.current_speed = static_cast<float>(current_state.vx);

        pub_control.publish(controlMsg);
    }
}

// =====================================================
//   STAN POJAZDU – ODOMETRIA
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
//   LOOKAHEAD MARKER
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
    lookahead_marker.color.g = 0.0;
    lookahead_marker.color.b = 0.0;

    pub_geo_marker.publish(lookahead_marker);
}

// =====================================================
//   PATH POINTS – KROPKI
// =====================================================
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
//   MODEL II RZĘDU – DYSKRETYZACJA ZOH
// =====================================================
void Controller::ZOH_for_steering()
{
    double max_speed = param_.get("model_max_steering_angle_rate"); // rad/s
    double dt = 1.0 / param_.get("odom_frequency");

    double old_delta   = mpc_state.delta;
    double old_d_delta = mpc_state.d_delta;

    double exact_delta   = Ad_maxon(0,0) * old_delta + Ad_maxon(0,1) * old_d_delta + Bd_maxon(0) * mpc_state.delta_request;
    double exact_d_delta = Ad_maxon(1,0) * old_delta + Ad_maxon(1,1) * old_d_delta + Bd_maxon(1) * mpc_state.delta_request;

    if (std::abs(exact_d_delta) <= max_speed)
    {
        mpc_state.delta   = exact_delta;
        mpc_state.d_delta = exact_d_delta;
    }
    else
    {
        double dir = (exact_d_delta > 0) ? 1.0 : -1.0;

        if (std::abs(old_d_delta) >= max_speed)
        {
            mpc_state.d_delta = dir * max_speed;
            mpc_state.delta   = old_delta + mpc_state.d_delta * dt;
        }
        else
        {
            double fraction = (max_speed - std::abs(old_d_delta)) /
                              (std::abs(exact_d_delta) - std::abs(old_d_delta));

            double t_accel = fraction * dt;
            double t_const = (1.0 - fraction) * dt;

            double dist_accel = dir * ( (std::abs(old_d_delta) + max_speed) / 2.0 ) * t_accel;
            double dist_const = dir * max_speed * t_const;

            mpc_state.delta   = old_delta + dist_accel + dist_const;
            mpc_state.d_delta = dir * max_speed;
        }
    }

    mpc_state.delta = std::clamp(mpc_state.delta, param_.get("min_delta"), param_.get("max_delta"));
}

double Controller::acc_to_throttle_percentage(double a_desired)
{
    double vehicle_mass = param_.get("model_m");
    double wheel_radius = param_.get("model_wheel_radius");
    double max_motor_torque = 4 * param_.get("model_max_motor_torque");

    const double drag_force = param_.get("model_Cd") * current_state.vx * current_state.vx +
                              param_.get("model_Cr0");
    (void)drag_force;

    double F_required = vehicle_mass * a_desired + drag_force - vehicle_mass * current_state.vy * current_state.yaw_rate;
    double torque_required = F_required * wheel_radius;

    double throttle_command = torque_required / max_motor_torque * 100.0;
    throttle_command = std::clamp(throttle_command, -100.0, 100.0);

    current_state.acc = a_desired;
    return throttle_command;
}

static inline double wrap_to_pi(double a)
{
    while (a >  M_PI) a -= 2.0*M_PI;
    while (a < -M_PI) a += 2.0*M_PI;
    return a;
}

void Controller::convert_state_to_mpc_state()
{
    if (!spline_complete_path.valid()) {
        ROS_WARN_STREAM("[MPC] Spline not valid, cannot convert state properly.");
        return;
    }

    v2_control::Vec2 q(current_state.X, current_state.Y);
    double s_proj = spline_complete_path.projectToSpline(q);
    last_s0 = s_proj;

    double path_yaw = spline_complete_path.getYaw(s_proj);

    double epsi = current_state.yaw - path_yaw;
    epsi = wrap_to_pi(epsi);

    double ey = spline_complete_path.signedNormalDistance(q, s_proj);

    mpc_state.ey   = ey;
    mpc_state.epsi = epsi;
    mpc_state.vy   = current_state.vy;
    mpc_state.r    = current_state.yaw_rate;
    // mpc_state.delta, mpc_state.d_delta, mpc_state.delta_request updated in ZOH_for_steering
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

    double max_ax = param_.get("model_mux") * 9.81;
    double max_ay = param_.get("model_muy") * 9.81;

    for (int i = 0; i <= 100; ++i) {
        double theta = (i / 100.0) * 2.0 * M_PI;
        geometry_msgs::Point p;
        p.x = max_ay * std::cos(theta);
        p.y = max_ax * std::sin(theta);
        p.z = 0.0;
        msg.points.push_back(p);
    }

    pub_gg_marker.publish(msg);
}

} // namespace v2_control