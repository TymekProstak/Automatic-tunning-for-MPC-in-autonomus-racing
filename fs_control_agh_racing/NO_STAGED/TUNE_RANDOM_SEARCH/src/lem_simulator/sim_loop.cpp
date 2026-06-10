#include "sim_loop.hpp"
#include <iostream>   // <<< diagnostyka
#include <exception>  // <<< diagnostyka
#include <algorithm>  // std::clamp
#include <sstream>
#include <iomanip>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_broadcaster.h>
#include <dv_interfaces/MPCDebug.h>
#include <dv_interfaces/RackAngleSensor.h>
#include <std_msgs/Float64.h>
#include "spline.hpp"

namespace lem_dynamics_sim_ {

    static inline int clamp_phase(int period, int phase)
    {
        if (period <= 1) return 0;
        phase %= period;
        if (phase < 0) phase += period;
        return phase;
    }
    
    static inline bool is_due(int step, int period, int phase)
    {
        if (period <= 0) return false;
        phase = clamp_phase(period, phase);
        return (step % period) == phase;
    }
    
    int Simulation_lem_ros_node::pick_phase_(int period)
    {
        if (period <= 1) return 0;
        std::uniform_int_distribution<int> dist(0, period - 1);
        return dist(phase_rng_);
    }

    static inline void update_top_abs(std::vector<double>& v, double x, std::size_t N = 10)
    {
        const double ax = std::abs(x);

        // jeśli jeszcze nie ma N elementów -> wrzuć i posortuj
        if (v.size() < N) {
            v.push_back(x);
            std::sort(v.begin(), v.end(),
                    [](double a, double b){ return std::abs(a) > std::abs(b); });
            return;
        }

        // jeśli x nie jest większe niż najmniejsze z TOP -> ignore
        double smallest_abs = std::abs(v.back());
        if (ax <= smallest_abs) return;

        // podmień najmniejsze i ponownie posortuj
        v.back() = x;
        std::sort(v.begin(), v.end(),
                [](double a, double b){ return std::abs(a) > std::abs(b); });
    }

    


using json = nlohmann::json;

// ====== Konstruktor / inicjalizacja ======
Simulation_lem_ros_node::Simulation_lem_ros_node(ros::NodeHandle& nh,
                                                 const std::string& param_file,
                                                 const std::string& cones_file,
                                                 const std::string& log_file )
{
    // --- DIAG: wczytywanie parametrów ---
    std::cout << "[INIT] Opening param file: " << param_file << std::endl;
    {
        try {
            std::ifstream f(param_file);
            if (!f.is_open()) {
                std::cerr << "[INIT][FAIL] Cannot open param file." << std::endl;
                throw std::runtime_error("Nie mogę otworzyć pliku parametrów: " + param_file);
            }
            std::cout << "[INIT] Param file opened. Parsing JSON..." << std::endl;

            nlohmann::json J;
            f >> J;
            std::cout << "[INIT] JSON parsed. Building ParamBank..." << std::endl;

            P_ = build_param_bank(J);
            std::cout << "[INIT][OK] ParamBank built. Count=" << P_.size() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[INIT][FAIL] Parameter loading failed. what(): " << e.what() << std::endl;
            throw; // nie zmieniamy logiki — dalej rzucamy wyjątek
        }
    }

    // --- DIAG: reset stanu ---
    std::cout << "[INIT] Resetting simulation state..." << std::endl;
    state_.setZero();


    // --- DIAG: wczytanie pachołków ---
    std::cout << "[INIT] Loading cones file: " << cones_file << std::endl;
    try {
        track_global_ = load_track_from_csv(cones_file);
        std::cout << "[INIT][OK] Cones loaded. cones=" << track_global_.cones.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INIT][FAIL] Cones load failed. what(): " << e.what() << std::endl;
        throw;
    }

    // --- Wczytywanie track_file do splajnu ---
    std::string track_file;
    if (nh.getParam("track_file", track_file)) {
        std::cout << "[INIT] Loading track file for spline: " << track_file << std::endl;
        std::vector<v2_control::Vec2> pts;
        std::ifstream in(track_file);
        std::string line;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            std::string normalized = line;
            for (char& c : normalized) if (c == ';') c = ',';
            std::stringstream ss(normalized);
            std::string token_x, token_y;
            if (!std::getline(ss, token_x, ',')) continue;
            if (!std::getline(ss, token_y, ',')) continue;
            try {
                pts.push_back(v2_control::Vec2(std::stod(token_x), std::stod(token_y)));
            } catch (...) {}
        }
        if (pts.size() >= 3) {
            center_line_spline_.build(pts, true); // closed loop
            std::cout << "[INIT][OK] Spline built. Length=" << center_line_spline_.totalLength() << std::endl;
        } else {
            std::cerr << "[INIT][WARN] Not enough points for spline." << std::endl;
        }
    } else {
        std::cerr << "[INIT][WARN] No 'track_file' param found. Spline won't be built." << std::endl;
    }
    //state_.x = center_line_spline_.getX(0.0);
    //state_.y = center_line_spline_.getY(0.0);
    //state_.yaw = center_line_spline_.getYaw(0.0);
    //state_.vx = 2.0; //
    //state_.omega_rr = 2.0/0.195;
    //state_.omega_rl = 2.0/0.195;
    //state_.omega_fl = 2.0/0.195;
    //state_.omega_fr = 2.0/0.195;

    // 2) PID / sterowanie – reset
    std::cout << "[INIT] Resetting control/PIDs state..." << std::endl;

    ///


    // 3) Interwały krokowe (po wczytaniu P_)
    std::cout << "[INIT] Computing step intervals from parameters..." << std::endl;
    try {
        compute_step_intervals_from_params_();
        std::cout << "[INIT][OK] Step intervals computed." << std::endl;

        // 6) Fazy (rozstrzelenie w obrębie okresu)
    phase_camera_shoot_          = pick_phase_(step_of_camera_shoot_);
    phase_wheel_encoder_reading_ = pick_phase_(step_of_wheel_encoder_reading_);
    phase_ins_reading_           = pick_phase_(step_of_ins_reading_);
    phase_mcu_reading_           = pick_phase_(step_mcu_reading_);

    // READ / SEND
    phase_torque_apply_ = pick_phase_(step_of_control_input_read_);
    phase_steer_apply_  = pick_phase_(step_of_steer_input_sending_);
    phase_gps_speed_reading_    = pick_phase_(step_gps_speed_reading_);
    phase_control_input_read_    = pick_phase_(step_of_control_input_read_);


        std::cout << "[PHASES] cam="    << phase_camera_shoot_
                << " enc="            << phase_wheel_encoder_reading_
                << " ins="            << phase_ins_reading_
                << " torque_apply="   << phase_torque_apply_
                << " steer_apply="    << phase_steer_apply_
                << " gps_speed="      << phase_gps_speed_reading_
                << " control_read="   << phase_control_input_read_
          << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[INIT][FAIL] compute_step_intervals_from_params_ failed. what(): " << e.what() << std::endl;
        throw;
    }

    // 4) ROS I/O 
    std::cout << "[INIT] Initializing ROS I/O (subscribers/publishers)..." << std::endl;
    try {
        sub_control_ = nh.subscribe<dv_interfaces::Control>(
            "/dv_board/control", 1, &Simulation_lem_ros_node::dv_control_callback, this, ros::TransportHints().tcpNoDelay());
        sub_mpc_debug_ = nh.subscribe<dv_interfaces::MPCDebug>(
            "/control/mpc_debug", 1, &Simulation_lem_ros_node::mpc_debug_callback_, this);
        pub_ins_   = nh.advertise<nav_msgs::Odometry>("/ins/pose", 1);
        pub_rack_angle_   = nh.advertise<dv_interfaces::RackAngleSensor>("/sensors/rack_angle", 1);
        pub_cones_ = nh.advertise<dv_interfaces::Cones>("/dv_cone_detector/cones", 1);
        pub_markers_cones_gt_  = nh.advertise<visualization_msgs::MarkerArray>("/viz/cones_gt", 1,true);
        pub_markers_cones_vis_ = nh.advertise<visualization_msgs::MarkerArray>("/viz/cones_vis", 1);
        pub_log_full_ = nh.advertise<dv_interfaces::full_state>("/debug/full_log_info", 1);
        pub_marker_bolid_ = nh.advertise<visualization_msgs::Marker>("/viz/bolide_model", 1);
        pub_gg_sphere_marker_ = nh.advertise<visualization_msgs::Marker>("/simulation/gg_sphere", 1);
        std::cout << "[INIT][OK] ROS I/O ready." << std::endl;
        
        auto check_pub = [&](const char* name, const ros::Publisher& p){
            if (!p) {
                std::cerr << "[INIT][FAIL] Publisher invalid right after advertise(): " << name << std::endl;
                ROS_ERROR("Publisher invalid right after advertise(): %s", name);
            }
        };
        check_pub("pub_ins_",               pub_ins_);
        check_pub("pub_cones_",             pub_cones_);
        check_pub("pub_markers_cones_gt_",  pub_markers_cones_gt_);
        check_pub("pub_markers_cones_vis_", pub_markers_cones_vis_);

    } catch (const std::exception& e) {
        std::cerr << "[INIT][FAIL] ROS I/O init failed. what(): " << e.what() << std::endl;
        throw;
    }

    // 5) publikacja conów z toru (ground truth) o wiecznym life-time
    std::cout << "[INIT] Publishing GT cones markers..." << std::endl;
    try {
        publish_cones_gt_markers_();
        std::cout << "[INIT][OK] GT cones published." << std::endl;
    } catch (const std::exception& e) {
        // nie przerywamy — to tylko markery
        std::cerr << "[INIT][WARN] publish_cones_gt_markers_ failed. what(): " << e.what() << std::endl;
    }

    // 6) Inicjalizacja logowania metryk jazdy
    if (!log_file.empty()) {
        metrics_log_file_path_ = log_file;
        const auto pos = metrics_log_file_path_.rfind(".csv");
        if (pos != std::string::npos) {
            metrics_log_file_path_.replace(pos, 4, "_metrics.csv");
        } else {
            metrics_log_file_path_ += "_metrics.csv";
        }
    } else {
        metrics_log_file_path_.clear(); // metryki wyłączone jeśli log_file pusty
    }

   

    // 7) Kolejka kamery pusta na start
    camera_queue_.clear();
    std::cout << "[INIT][DONE] Node constructed successfully." << std::endl;

    ROS_WARN_STREAM("Init yaw=" << state_.yaw << " vy=" << state_.vy);

    // --- Traction control PID init (4WD: 8x PID) ---
    {
        PIDParams drive{};
        drive.Kp = P_.get("pid_traction_p");
        drive.Ki = P_.get("pid_traction_i");
        drive.Kd = P_.get("pid_traction_d");

        drive.saturation_upper = P_.get("pid_traction_max_drive");
        drive.saturation_lower = P_.get("pid_traction_min_drive");

        drive.anti_windup_gain = P_.get("pid_traction_anti_windup_gain_drive");
        drive.leak_time_scale  = P_.get("pid_traction_leak_time_scale_drive");

        PIDParams brake{};
        brake.Kp = P_.get("pid_traction_p");
        brake.Ki = P_.get("pid_traction_i");
        brake.Kd = P_.get("pid_traction_d");

        brake.saturation_upper = P_.get("pid_traction_max_brake");
        brake.saturation_lower = P_.get("pid_traction_min_brake");

        brake.anti_windup_gain = P_.get("pid_traction_anti_windup_gain_brake");
        brake.leak_time_scale  = P_.get("pid_traction_leak_time_scale_brake");

        // DRIVE PIDs
        tc_drive_fl_.set_params(drive); tc_drive_fl_.reset();
        tc_drive_fr_.set_params(drive); tc_drive_fr_.reset();
        tc_drive_rl_.set_params(drive); tc_drive_rl_.reset();
        tc_drive_rr_.set_params(drive); tc_drive_rr_.reset();

        // BRAKE PIDs
        tc_brake_fl_.set_params(brake); tc_brake_fl_.reset();
        tc_brake_fr_.set_params(brake); tc_brake_fr_.reset();
        tc_brake_rl_.set_params(brake); tc_brake_rl_.reset();
        tc_brake_rr_.set_params(brake); tc_brake_rr_.reset();
        
        // FX PID
        PIDParams fx_params{};
        fx_params.Kp = P_.get("pid_fx_target_p");
        fx_params.Ki = P_.get("pid_fx_target_i");
        fx_params.Kd = P_.get("pid_fx_target_d");
        fx_params.saturation_upper = P_.get("pid_fx_target_max");
        fx_params.saturation_lower = P_.get("pid_fx_target_min");
        fx_params.anti_windup_gain = P_.get("pid_fx_target_anti_windup_gain");
        fx_params.leak_time_scale  = P_.get("pid_fx_target_leak_time_scale");
        pid_fx_.set_params(fx_params); pid_fx_.reset();

        // MZ PID
        PIDParams mz_params{};
        mz_params.Kp = P_.get("pid_mz_target_p");
        mz_params.Ki = P_.get("pid_mz_target_i");
        mz_params.Kd = P_.get("pid_mz_target_d");
        mz_params.saturation_upper = P_.get("pid_mz_target_max");
        mz_params.saturation_lower = P_.get("pid_mz_target_min");
        mz_params.anti_windup_gain = P_.get("pid_mz_target_anti_windup_gain");
        mz_params.leak_time_scale  = P_.get("pid_mz_target_leak_time_scale");
        pid_mz_.set_params(mz_params); pid_mz_.reset();
    }


    // --- Kalman init (jeśli używany) ---
    {
        // KalmanFilter ma konstruktor z ParamBank; tworzymy i resetujemy go po wczytaniu P_
        kalman_filter_ = KalmanFilter(P_);

        // opcjonalnie: ins_mode z ROS param (gauss|kalman)
        // (jeśli param nie istnieje, zostaje domyślne z .hpp)
        std::string mode;
        if (nh.getParam("ins_mode", mode)) {
            ins_mode_ = mode;
        }
        bool use_llc;
        if (nh.getParam("low_level_controlers", use_llc)) {
            lov_level_control_on = use_llc;

        }
    }
    int sim_time_ ;
    if(nh.getParam("sim_time", sim_time_)){
        sim_time = sim_time_;
    }

    std::cout << "[INIT] sim_time = " << sim_time_
            << "  ( <0 => infinite , >=0 => stop after sim_time seconds )"
            << std::endl;

    ROS_WARN_STREAM("[INIT] sim_time=" << sim_time);

    if (!nh.getParam("critical_crash_multiplier", critical_crash_multiplier_)) {
        critical_crash_multiplier_ = 1.0;
    }
    
    ROS_WARN_STREAM("[INIT] critical_crash_multiplier=" << critical_crash_multiplier_);
}

// ====== Destruktor ======
Simulation_lem_ros_node::~Simulation_lem_ros_node() {
    std::cout << "[DTOR] Saving ride metrics..." << std::endl;
    log_metric_of_ride_data_();

}

// ====== Interfejs publiczny ======
void Simulation_lem_ros_node::step() {
    try {
        const auto t_step_start = std::chrono::steady_clock::now();

        if (crashed_ && !shutdown_requested_)
        {
            shutdown_requested_ = true;
            ROS_ERROR_STREAM("[CRASH] " << crash_reason_
                             << " | step=" << crash_step_
                             << " | t=" << crash_time_s_ << " s"
                             << " -> shutting down ROS node.");
            ros::shutdown();
            return;
        }

        read_control_by_dv_board_if_due();
        read_wheel_encoder_if_due_();
        read_ins_if_due_();

        send_to_ts_if_due();
        send_steer_to_maxon_if_due_();

        publish_rack_angle_();

        const auto t_sim_start = std::chrono::steady_clock::now();

        euler_sim_timestep(
            state_,
            Input(torque_cmd_rr_, torque_cmd_rl_, torque_cmd_fr_, torque_cmd_fl_, steer_command_to_maxon_),
            P_
        );

        const auto t_sim_end = std::chrono::steady_clock::now();

        ++step_number_;

        pub_full_state_();
      publish_bolid_marker_();
      publish_bolid_tf_true();

        const auto t_step_end = std::chrono::steady_clock::now();

        const double sim_dt_s = P_.get("simulation_time_step");

        const double sim_wall_s =
            std::chrono::duration<double>(t_sim_end - t_sim_start).count();

        const double step_wall_s =
            std::chrono::duration<double>(t_step_end - t_step_start).count();

        const double sim_wall_ms  = 1000.0 * sim_wall_s;
        const double step_wall_ms = 1000.0 * step_wall_s;
        const double budget_ms    = 1000.0 * sim_dt_s;

        // wypisuj tylko gdy przekroczono budżet kroku
        if (step_wall_s > sim_dt_s)
        {
            ROS_WARN_STREAM(
                "[STEP_TIMING][OVERRUN] "
                << "step=" << step_number_
                << " | total_step=" << step_wall_ms << " ms"
                << " | sim_only="   << sim_wall_ms  << " ms"
                << " | budget="     << budget_ms    << " ms"
                << " | overrun="    << (step_wall_ms - budget_ms) << " ms"
            );
        }

        // // opcjonalnie: rzadszy log informacyjny co 100 kroków
        // if (step_number_ % 100 == 0)
        // {
        //     ROS_INFO_STREAM(
        //         "[STEP_TIMING] "
        //         << "step=" << step_number_
        //         << " | total_step=" << step_wall_ms << " ms"
        //         << " | sim_only="   << sim_wall_ms  << " ms"
        //         << " | budget="     << budget_ms    << " ms"
        //     );
        // }

        // ======================================================
        // STOP AFTER sim_time_ seconds (if enabled)
        // ======================================================
        if (sim_time >= 0)
        {
            const double dt = P_.get("simulation_time_step");
            const double t_sim = step_number_ * dt;

            if (t_sim >= static_cast<double>(sim_time))
            {
                ROS_WARN_STREAM("[SIM_LOOP] sim_time reached: " << t_sim
                                << " / " << sim_time << " s -> shutting down ROS node.");
                ros::shutdown();
                return;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[STEP][FAIL] Exception in simulation step. what(): " << e.what() << std::endl;
        throw;
    }
}


State Simulation_lem_ros_node::get_state() const { return state_; }
ParamBank Simulation_lem_ros_node::get_parameters() const { return P_; }
int Simulation_lem_ros_node::get_step_number() const { return step_number_; }

// ====== ROS callback ======
// caching last requested input from control
void Simulation_lem_ros_node::dv_control_callback(const dv_interfaces::Control::ConstPtr& msg)
{
    last_input_cached = *msg;
}

// ====== Pomocnicze ======
void Simulation_lem_ros_node::compute_step_intervals_from_params_() {
    const double dt = P_.get("simulation_time_step");
    std::cout << "[INTERVALS] dt=" << dt << std::endl;

    // sensor cadences
    step_of_camera_shoot_          = std::max(1, (int)std::round(1.0 / P_.get("frames_per_second") / dt));
    step_of_wheel_encoder_reading_ = std::max(1, (int)std::round(P_.get("wheel_encoder_reading_time_step") / dt));
    step_of_ins_reading_           = std::max(1, (int)std::round(1.0 / P_.get("ins_frequancy") / dt));
    step_gps_speed_reading_ = std::max(1, (int)std::round(1.0 / P_.get("gps_speed_frequancy") / dt));
    step_imu_reading_            = std::max(1, (int)std::round(1.0 / P_.get("acc_frequancy") / dt));
    step_mcu_reading_ = std::max(1, (int)std::round(1.0 / 60.0 / dt)); // 60Hz rack angle
   

    // READ/SEND pipeline cadences (must exist in .hpp)
    step_of_control_input_read_       = std::max(1, (int)std::round(P_.get("control_to_dv_boad_read_time_step") / dt));
    step_of_steer_input_sending_      = std::max(1, (int)std::round(P_.get("dv_board_to_maxon_time_step") / dt));
    step_number_torque_input_sending_ = std::max(1, (int)std::round(P_.get("dv_board_tractive_system_time_step") / dt));

    std::cout << "[INTERVALS] cam="      << step_of_camera_shoot_
              << " enc="                << step_of_wheel_encoder_reading_
              << " ins="                << step_of_ins_reading_
              << " gps="                << step_gps_speed_reading_
              << " ctrl_read="          << step_of_control_input_read_
              << " steer_send="         << step_of_steer_input_sending_
              << " torque_send="        << step_number_torque_input_sending_
              << std::endl;
}

// dv board reads control topic at fixed cadence
void Simulation_lem_ros_node::read_control_by_dv_board_if_due()
{
    if( is_due(step_number_, step_of_control_input_read_, phase_control_input_read_) )
    last_input_read_by_dv_board = last_input_cached;
}

void Simulation_lem_ros_node::read_wheel_encoder_if_due_()
{
    if (step_of_wheel_encoder_reading_ <= 0) return;
    if (!is_due(step_number_, step_of_wheel_encoder_reading_, phase_wheel_encoder_reading_)) return;

    const double R = P_.get("R");

    wheel_speed_fl_ = state_.omega_fl * R;
    wheel_speed_fr_ = state_.omega_fr * R;
    wheel_speed_rl_ = state_.omega_rl * R;
    wheel_speed_rr_ = state_.omega_rr * R;
}

void Simulation_lem_ros_node::read_steer_by_orin_if_due_()
{
    if( is_due(step_number_, step_of_steer_input_sending_, phase_steer_apply_) )
    {
        steer_command_to_maxon_ = last_input_cached.steeringAngle_rad;
    }


}
void Simulation_lem_ros_node::read_ins_if_due_()
{
    const bool due_ins = (step_of_ins_reading_ > 0) &&
                         is_due(step_number_, step_of_ins_reading_, phase_ins_reading_);

    const bool due_gps = (step_gps_speed_reading_ > 0) &&
                         is_due(step_number_, step_gps_speed_reading_, phase_gps_speed_reading_);

    const bool due_imu = (step_imu_reading_ > 0) &&
                         is_due(step_number_, step_imu_reading_, 0);

    const int calib_steps = std::max(
        1,
        static_cast<int>(P_.get("calibration_time") / P_.get("simulation_time_step"))
    );
    const bool calibrated = (step_number_ > calib_steps);

    // Jeśli nic nie robimy w tej iteracji – wyjdź
    if (!due_ins && !due_gps && !due_imu) return;


    // ----------------------------
    // IMU (cache)
    // ----------------------------
    
    if (due_imu) {
        ImuMeasurement imu_meas{};
    
        const int imu_period = std::max(1, step_imu_reading_);
        const double dt_imu = P_.get("simulation_time_step") * imu_period;
    
        if (!std::isfinite(dt_imu) || dt_imu <= 0.0) {
            ROS_ERROR_STREAM("[IMU] dt_imu invalid: " << dt_imu
                             << " (sim_dt=" << P_.get("simulation_time_step")
                             << ", step_imu_reading_=" << step_imu_reading_ << ")");
            has_last_imu_ = false;
            return;
        }
    
        auto finite_or = [](double v, double fb){
            return std::isfinite(v) ? v : fb;
        };
    
        const double prev_ax = finite_or(state_.prev_ax, 0.0);
        const double prev_ay = finite_or(state_.prev_ay, 0.0);
        const double yaw_rate = finite_or(state_.yaw_rate, 0.0);
    
        // random walk bias update
        sim_b_g  = finite_or(sim_b_g,  0.0);
        sim_b_ax = finite_or(sim_b_ax, 0.0);
        sim_b_ay = finite_or(sim_b_ay, 0.0);
    
        sim_b_g  += P_.get("gyro_bias_rw") * std::sqrt(dt_imu) * random_noise_generator_();
        sim_b_ax += P_.get("acc_bias_rw")  * std::sqrt(dt_imu) * random_noise_generator_();
        sim_b_ay += P_.get("acc_bias_rw")  * std::sqrt(dt_imu) * random_noise_generator_();
    
        imu_meas.yaw_rate = yaw_rate + sim_b_g
                            + P_.get("gyro_noise_std") * random_noise_generator_();
        imu_meas.ax = prev_ax + sim_b_ax
                    + P_.get("acc_noise_std") * random_noise_generator_();
        imu_meas.ay = prev_ay + sim_b_ay
                    + P_.get("acc_noise_std") * random_noise_generator_();
    
        if (!std::isfinite(imu_meas.yaw_rate) || !std::isfinite(imu_meas.ax) || !std::isfinite(imu_meas.ay)) {
            ROS_ERROR_STREAM("[IMU] NaN produced: yaw_rate=" << imu_meas.yaw_rate
                             << " ax=" << imu_meas.ax << " ay=" << imu_meas.ay
                             << " | dt_imu=" << dt_imu
                             << " | prev_ax=" << state_.prev_ax << " prev_ay=" << state_.prev_ay
                             << " | sim_b_g=" << sim_b_g << " sim_b_ax=" << sim_b_ax << " sim_b_ay=" << sim_b_ay);
            has_last_imu_ = false;
            return;
        }
    
        last_imu_ = imu_meas;
        has_last_imu_ = true;
    
        if (ins_mode_ == "kalman") {
            kalman_filter_.predict(imu_meas, dt_imu);
        }

        // ----------------------------------------------------------
        // IMU dead-reckoning offset: integrate body-frame accel
        // between GPS samples to get smooth velocity for INS output.
        // ax_imu, ay_imu are in BODY frame already.
        // ----------------------------------------------------------
        const double vx_body_est = gps_vx_body_ + imu_vx_offset_;
        const double vy_body_est = gps_vy_body_ + imu_vy_offset_;
        
        const double dvx = imu_meas.ax + imu_meas.yaw_rate * vy_body_est;
        const double dvy = imu_meas.ay - imu_meas.yaw_rate * vx_body_est;
        
        imu_vx_offset_ += dvx * dt_imu;
        imu_vy_offset_ += dvy * dt_imu;
    }
    

    // ----------------------------
    // GPS (cache) + INS velocity from GPS ALWAYS
    // ----------------------------
    if (due_gps) {
        GpsMeasurement gps_meas{};

        const double vx_global = state_.vx * std::cos(state_.yaw) - state_.vy * std::sin(state_.yaw);
        const double vy_global = state_.vx * std::sin(state_.yaw) + state_.vy * std::cos(state_.yaw);

        gps_meas.x   = state_.x + P_.get("gps_position_noise") * random_noise_generator_();
        gps_meas.y   = state_.y + P_.get("gps_position_noise") * random_noise_generator_();
        gps_meas.vx  = vx_global + P_.get("gps_speed_noise")   * random_noise_generator_();
        gps_meas.vy  = vy_global + P_.get("gps_speed_noise")   * random_noise_generator_();
        double yaw = state_.yaw + P_.get("gps_yaw_noise") * random_noise_generator_();
        unwrap_angle(yaw);
        gps_meas.yaw = yaw;
        
        

        last_gps_ = gps_meas;
        has_last_gps_ = true;

        // GPS gives global-frame velocities → rotate to body frame
        // using the best yaw estimate we have (GPS yaw with noise)
        const double cos_yaw = std::cos(gps_meas.yaw);
        const double sin_yaw = std::sin(gps_meas.yaw);
        gps_vx_body_ =  cos_yaw * gps_meas.vx + sin_yaw * gps_meas.vy;
        gps_vy_body_ = -sin_yaw * gps_meas.vx + cos_yaw * gps_meas.vy;

        // Reset IMU dead-reckoning offset on each new GPS sample
        imu_vx_offset_ = 0.0;
        imu_vy_offset_ = 0.0;

    
        if (ins_mode_ == "kalman") {
            
            kalman_filter_.update_gps(gps_meas);

        }
        
    }

    // ----------------------------
    // INS publish tick
    // ----------------------------
    if (!due_ins) return;

    
        if (!has_last_gps_) {
           
              // jeszcze nie przyszedł żaden GPS
            ins_data_to_be_published_.vx = 0.0;
            ins_data_to_be_published_.vy = 0.0;
        }


    if (ins_mode_ == "gauss") {
        ins_data_to_be_published_.x   = state_.x + P_.get("pose_noise") * random_noise_generator_();
        ins_data_to_be_published_.y   = state_.y + P_.get("pose_noise") * random_noise_generator_();
        double yaw = state_.yaw + P_.get("orientation_noise") * random_noise_generator_();
        unwrap_angle(yaw);
        ins_data_to_be_published_.yaw = yaw;
        ins_data_to_be_published_.yaw_rate = state_.yaw_rate +P_.get("rotation_noise") * random_noise_generator_();

        // Velocity: GPS base + IMU dead-reckoning offset (body frame)
        if (has_last_gps_) {
            double vx_body = gps_vx_body_ + imu_vx_offset_;
            double vy_body = gps_vy_body_ + imu_vy_offset_;
            
            // Zakładam, że ins_data_to_be_published_.yaw zostało ustawione wyżej dla tego trybu
            ins_data_to_be_published_.vx = vx_body * std::cos(ins_data_to_be_published_.yaw) - vy_body * std::sin(ins_data_to_be_published_.yaw);
            ins_data_to_be_published_.vy = vx_body * std::sin(ins_data_to_be_published_.yaw) + vy_body * std::cos(ins_data_to_be_published_.yaw);
        }
    }  else if (ins_mode_ == "kalman") {
 
        ins_data_to_be_published_.x = kalman_filter_.get_state().x;
        ins_data_to_be_published_.y = kalman_filter_.get_state().y;
        ins_data_to_be_published_.yaw = kalman_filter_.get_state().yaw;
        ins_data_to_be_published_.yaw_rate = kalman_filter_.get_state().yaw_rate;

        // Velocity: GPS base + IMU dead-reckoning offset (body frame)
        if (has_last_gps_) {
            double vx_body = gps_vx_body_ + imu_vx_offset_;
            double vy_body = gps_vy_body_ + imu_vy_offset_;
            
            double current_yaw = kalman_filter_.get_state().yaw;
            
            ins_data_to_be_published_.vx = vx_body * std::cos(current_yaw) - vy_body * std::sin(current_yaw);
            ins_data_to_be_published_.vy = vx_body * std::sin(current_yaw) + vy_body * std::cos(current_yaw);
        }

    } 

    if (calibrated) publish_ins_(ins_data_to_be_published_);
    last_ins_data_already_published_ = ins_data_to_be_published_;
    if (calibrated) publish_bolid_tf_ins(ins_data_to_be_published_);
}

void Simulation_lem_ros_node::shoot_camera_or_enqueue_if_due_()
{
    if (step_of_camera_shoot_ <= 0) return;
    if (!is_due(step_number_, step_of_camera_shoot_, phase_camera_shoot_)) return;

    Track detection = shoot_a_frame(track_global_, P_, state_);
    const double dt = P_.get("simulation_time_step");
    const double vision_exec_time = sample_vision_exec_time_();
    const int processing_steps = std::max(0, (int)std::round(vision_exec_time / dt));

    CameraTask task;
    task.ready_step = step_number_ + processing_steps;
    task.frame      = std::move(detection);

    if ((int)camera_queue_.size() >= 3) {
        camera_queue_.pop_front();
        timestamp_queue_.pop_front();
    }

    camera_queue_.push_back(std::move(task));
    timestamp_queue_.push_back(ros::Time::now());
}


void Simulation_lem_ros_node::publish_ready_camera_frames_from_queue_() {
    while (!camera_queue_.empty() && camera_queue_.front().ready_step <= step_number_) {
        const auto& task = camera_queue_.front();
        const auto& timestamp = timestamp_queue_.front();
        publish_cones_(task.frame, timestamp);
        publish_cones_vision_markers_(task.frame, timestamp);
        camera_queue_.pop_front();
        timestamp_queue_.pop_front();
    }
}

// sending torque to tractive system if due - dv_board communicates with tractive system at fixed cadence
void Simulation_lem_ros_node::send_to_ts_if_due()
{
    if (!is_due(step_number_, step_number_torque_input_sending_, phase_torque_apply_)) {
        return;
    }

    const bool speed_mode = last_input_read_by_dv_board.move_type; // 0: torque%, 1: speed
    const double Ts = P_.get("simulation_time_step") * step_number_torque_input_sending_;

    // =========================================================
    // LLC OFF -> "na pałę" (bez PID(Fx/Mz), bez QP alloc, bez TC)
    // =========================================================
    if (!lov_level_control_on)
    {
        if (!speed_mode)
        {
            // FF: fx/mz -> "na pałę" torque vectoring
            const double fx_target_ff = last_input_read_by_dv_board.fx_target;  // [N]
            const double mz_target_ff = last_input_read_by_dv_board.mz_target;  // [Nm]

            const double R = P_.get("R");        // [m]
            const double t_front = P_.get("t_front");  // [m]
            const double t_rear  = P_.get("t_rear");   // [m]

            const double F_base = 0.25 * fx_target_ff; // [N] na koło

            const double t_sum = std::max(1e-6, t_front + t_rear);
            const double mz_front = mz_target_ff * (t_front / t_sum);
            const double mz_rear  = mz_target_ff * (t_rear  / t_sum);

            const double dF_front = mz_front / std::max(1e-6, t_front); // [N]
            const double dF_rear  = mz_rear  / std::max(1e-6, t_rear);  // [N]

            const double F_fl = F_base - dF_front;
            const double F_fr = F_base + dF_front;
            const double F_rl = F_base - dF_rear;
            const double F_rr = F_base + dF_rear;

            torque_cmd_fl_ = F_fl * R;
            torque_cmd_fr_ = F_fr * R;
            torque_cmd_rl_ = F_rl * R;
            torque_cmd_rr_ = F_rr * R;


            

            const double TMAX = P_.get("max_torque");
            torque_cmd_fl_ = std::clamp(torque_cmd_fl_, -TMAX, TMAX);
            torque_cmd_fr_ = std::clamp(torque_cmd_fr_, -TMAX, TMAX);
            torque_cmd_rl_ = std::clamp(torque_cmd_rl_, -TMAX, TMAX);
            torque_cmd_rr_ = std::clamp(torque_cmd_rr_, -TMAX, TMAX);

            return;
        }
        else
        {
            // SPEED MODE: PID prędkości -> po równo (bez TC)
            const double wheel_speed_avg =
                0.25 * (wheel_speed_fl_ + wheel_speed_fr_ + wheel_speed_rl_ + wheel_speed_rr_);

            const double error =
                static_cast<double>(last_input_read_by_dv_board.movement) - wheel_speed_avg;

            prev_I_speed_pid_ += (error + prev_error_speed_pid_) * 0.5 * Ts;

            double u_pid =
                P_.get("pid_speed_p") * error +
                P_.get("pid_speed_i") * prev_I_speed_pid_ +
                P_.get("pid_speed_d") * (error - prev_error_speed_pid_) / Ts;

            prev_error_speed_pid_ = error;

            u_pid = std::clamp(u_pid, P_.get("pid_speed_min"), P_.get("pid_speed_max"));

            const double torque_total =
                (u_pid * 4.0 * P_.get("max_torque")) / P_.get("pid_speed_scale");

            torque_cmd_fl_ = 0.25 * torque_total;
            torque_cmd_fr_ = 0.25 * torque_total;
            torque_cmd_rl_ = 0.25 * torque_total;
            torque_cmd_rr_ = 0.25 * torque_total;

            return;
        }
    }

    // =========================================================
    // LLC ON -> FF + PID(Fx/Mz) + allocator + TC(8 PID)
    // =========================================================

    if (!speed_mode)
    {
        // FF zawsze
        const double fx_target_ff = last_input_read_by_dv_board.fx_target;
        const double mz_target_ff = last_input_read_by_dv_board.mz_target;

        double fx_target = fx_target_ff;
        double mz_target = mz_target_ff;

        // PID tylko gdy mamy IMU
        if (has_last_imu_)
        {
            const double ax_target = last_input_read_by_dv_board.ax_target;
            const double next_yaw_rate_target = last_input_read_by_dv_board.next_yaw_rate_target;

            // std::cout << "[INPUT] ax_target=" << ax_target
            //           << " next_yaw_rate_target=" << next_yaw_rate_target
            //           << std::endl;
            // std::cout << "[SENSOR] ax=" << last_imu_.ax
            //           << " yaw_rate=" << last_imu_.yaw_rate
            //           << std::endl;
            const double ax_error       = ax_target - (last_imu_.ax + (gps_vy_body_ + imu_vy_offset_)*last_imu_.yaw_rate); // dodaj efekt odśrodkowy z obrotu (yaw_rate * v_y)
            const double yaw_rate_error = next_yaw_rate_target - last_imu_.yaw_rate;

            pid_fx_.update(ax_error, Ts, true);
            pid_mz_.update(yaw_rate_error, Ts, true);

            fx_target += pid_fx_.get_output();
            mz_target += pid_mz_.get_output();
            // std::cout << "[PID] ax_error=" << ax_error
            //           << " yaw_rate_error=" << yaw_rate_error
            //           << " pid_fx_out=" << pid_fx_.get_output()
            //           << " pid_mz_out=" << pid_mz_.get_output()
            //           << std::endl;
        }

        Torque_allocation alloc = allocate_torque_optimaly(fx_target, mz_target);
        torque_cmd_fl_ = alloc.torque_fl;
        torque_cmd_fr_ = alloc.torque_fr;
        torque_cmd_rl_ = alloc.torque_rl;
        torque_cmd_rr_ = alloc.torque_rr;
        // torque_cmd_fl_ = 0.25 * fx_target_ff * P_.get("R");
        // torque_cmd_fr_ = 0.25 * fx_target_ff * P_.get("R");
        // torque_cmd_rl_ = 0.25 * fx_target_ff * P_.get("R");
        // torque_cmd_rr_ = 0.25 * fx_target_ff * P_.get("R");
       
        // TC (8 PID)
        apply_traction_control_4wd_(Ts);

        const double TMAX = P_.get("max_torque");
        torque_cmd_fl_ = std::clamp(torque_cmd_fl_, -TMAX, TMAX);
        torque_cmd_fr_ = std::clamp(torque_cmd_fr_, -TMAX, TMAX);
        torque_cmd_rl_ = std::clamp(torque_cmd_rl_, -TMAX, TMAX);
        torque_cmd_rr_ = std::clamp(torque_cmd_rr_, -TMAX, TMAX);

        return;
    }
    else
    {
        // SPEED MODE -> po równo + TC (jeśli chcesz; zostawiam bo tak miałeś)
        const double wheel_speed_avg =
            0.25 * (wheel_speed_fl_ + wheel_speed_fr_ + wheel_speed_rl_ + wheel_speed_rr_);

        const double error =
            static_cast<double>(last_input_read_by_dv_board.movement) - wheel_speed_avg;

        prev_I_speed_pid_ += (error + prev_error_speed_pid_) * 0.5 * Ts;

        double u_pid =
            P_.get("pid_speed_p") * error +
            P_.get("pid_speed_i") * prev_I_speed_pid_ +
            P_.get("pid_speed_d") * (error - prev_error_speed_pid_) / Ts;

        prev_error_speed_pid_ = error;

        u_pid = std::clamp(u_pid, P_.get("pid_speed_min"), P_.get("pid_speed_max"));

        const double torque_total =
            (u_pid * 4.0 * P_.get("max_torque")) / P_.get("pid_speed_scale");

        torque_cmd_fl_ = 0.25 * torque_total;
        torque_cmd_fr_ = 0.25 * torque_total;
        torque_cmd_rl_ = 0.25 * torque_total;
        torque_cmd_rr_ = 0.25 * torque_total;

        apply_traction_control_4wd_(Ts);

        const double TMAX = P_.get("max_torque");
        torque_cmd_fl_ = std::clamp(torque_cmd_fl_, -TMAX, TMAX);
        torque_cmd_fr_ = std::clamp(torque_cmd_fr_, -TMAX, TMAX);
        torque_cmd_rl_ = std::clamp(torque_cmd_rl_, -TMAX, TMAX);
        torque_cmd_rr_ = std::clamp(torque_cmd_rr_, -TMAX, TMAX);

        return;
    }
}

double Simulation_lem_ros_node::random_noise_generator_() const {
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::normal_distribution<double> N01(0.0, 1.0);
    return N01(rng);
}

void Simulation_lem_ros_node::publish_ins_(const INS_data& ins){
    nav_msgs::Odometry odom_msg{};
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = "map";
    odom_msg.child_frame_id  = "bolide_CoG";

    // --- TEST MODE: publish simulation ground-truth instead of INS sample ---
    // (original INS-based assignments kept commented for reference)
    
    odom_msg.pose.pose.position.x = ins.x;
    odom_msg.pose.pose.position.y = ins.y;
    odom_msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, ins.yaw);
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();

    odom_msg.twist.twist.linear.x  = ins.vx;
    odom_msg.twist.twist.linear.y  = ins.vy;
    odom_msg.twist.twist.angular.z = ins.yaw_rate;
    

    // // Use simulation ground-truth state_ for testing
    // odom_msg.pose.pose.position.x = state_.x;
    // odom_msg.pose.pose.position.y = state_.y;
    // odom_msg.pose.pose.position.z = 0.0;

    // tf2::Quaternion q;
    // q.setRPY(0.0, 0.0, state_.yaw);
    // odom_msg.pose.pose.orientation.x = q.x();
    // odom_msg.pose.pose.orientation.y = q.y();
    // odom_msg.pose.pose.orientation.z = q.z();
    // odom_msg.pose.pose.orientation.w = q.w();

    // // publish body-frame velocities transformed to global frame using state_.yaw
    // odom_msg.twist.twist.linear.x  = state_.vx * std::cos(state_.yaw) - state_.vy * std::sin(state_.yaw);
    // odom_msg.twist.twist.linear.y  = state_.vx * std::sin(state_.yaw) + state_.vy * std::cos(state_.yaw);
    // odom_msg.twist.twist.angular.z = state_.yaw_rate;

    pub_ins_.publish(odom_msg);
}

void Simulation_lem_ros_node::publish_cones_(const Track& cones, ros::Time timestamp){
    if (!pub_cones_) {
        std::cerr << "[PUB][ERROR] pub_cones_ invalid; NOT publishing cones." << std::endl;
        ROS_ERROR("pub_cones_ invalid; not publishing cones");
        return;
    }
   
   
    dv_interfaces::Cones cones_msg;
    cones_msg.header.stamp = timestamp;
    cones_msg.header.frame_id = "camera_base";
    for (const auto& cone : cones.cones) {
        dv_interfaces::Cone cone_msg;
        cone_msg.confidence = 1.0;
        cone_msg.x = static_cast<float>(cone.x);
        cone_msg.y = static_cast<float>(cone.y);
        cone_msg.z = static_cast<float>(cone.z);
        cone_msg.distance_uncertainty = 0.0f;
        cone_msg.class_name = cone.color;
        cones_msg.cones.push_back(cone_msg);
    }
    pub_cones_.publish(cones_msg);
}

double Simulation_lem_ros_node::sample_vision_exec_time_() const {
    const double mu  = P_.get("mean_time_of_vision_execuction");
    const double var = P_.get("var_of_vision_time_execution");

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> normal(mu, std::sqrt(std::max(0.0, var)));

    const double exec_time = normal(rng);
    return std::max(exec_time, 0.0);
}


void Simulation_lem_ros_node::publish_cones_vision_markers_(
    const Track& det, const ros::Time& acquisition_stamp)
{
    if (!pub_markers_cones_vis_) {
        ROS_ERROR("pub_markers_cones_vis_ invalid; not publishing vision markers");
        return;
    }

    visualization_msgs::MarkerArray arr;

    // ======================================================
    // 1) Usuń poprzednie markery wizji (TYLKO SWOJE)
    // ======================================================
    for (int id = 200; id < 200 + last_frame_size_; id++)
    {
        visualization_msgs::Marker del;
        del.header.frame_id = "camera_base";
        del.header.stamp    = acquisition_stamp;
        del.ns              = "cones_vis";
        del.id              = id;
        del.action          = visualization_msgs::Marker::DELETE;
        arr.markers.push_back(del);
    }

    // ======================================================
    // 2) Dodaj nowe markery
    // ======================================================
    const double fps = std::max(1e-3, P_.get("frames_per_second"));
    const ros::Duration lifetime(5.0 / fps);

    int id = 200;
    for (const auto& c : det.cones)
    {
        std_msgs::ColorRGBA col = color_from_class_vision(c.color, 0.7f);

        visualization_msgs::Marker m;
        m.header.frame_id = "camera_base";
        m.header.stamp    = ros::Time::now();
        m.ns              = "cones_vis";
        m.id              = id++;
        m.type            = visualization_msgs::Marker::CUBE;
        m.action          = visualization_msgs::Marker::ADD;

        m.pose.position.x = c.x;
        m.pose.position.y = c.y;
        m.pose.position.z = c.z + 0.15;
        m.pose.orientation.w = 1.0;

        m.scale.x = 0.30;
        m.scale.y = 0.30;
        m.scale.z = 0.30;

        m.color    = col;
        m.lifetime = lifetime;
        m.frame_locked = false;

        arr.markers.push_back(std::move(m));
    }

    // ======================================================
    // 3) Zapisz ile markerów było w aktualnej ramce
    // ======================================================
    last_frame_size_ = det.cones.size();

    pub_markers_cones_vis_.publish(arr);
}


void Simulation_lem_ros_node::publish_bolid_tf_true() {
    geometry_msgs::TransformStamped tf_true;
    tf_true.header.stamp = ros::Time::now();
    tf_true.header.frame_id = "map";
    tf_true.child_frame_id  = "bolide_true";
    tf_true.transform.translation.x = state_.x;
    tf_true.transform.translation.y = state_.y;
    tf_true.transform.translation.z = 0.0;
    tf2::Quaternion q1; q1.setRPY(0, 0, state_.yaw);
    tf_true.transform.rotation.x = q1.x();
    tf_true.transform.rotation.y = q1.y();
    tf_true.transform.rotation.z = q1.z();
    tf_true.transform.rotation.w = q1.w();
    tf_br_.sendTransform(tf_true);
}




void Simulation_lem_ros_node::publish_bolid_tf_ins(const INS_data& ins){
    
    if (!pub_ins_) {
        std::cerr << "[PUB][ERROR] pub_ins_ invalid; NOT publishing INS." << std::endl;
        ROS_ERROR("pub_ins_ invalid; not publishing INS");
        return;
    }
    
    
    
    geometry_msgs::TransformStamped tf_ins;
    tf_ins.header.stamp = ros::Time::now();
    tf_ins.header.frame_id = "map";
    tf_ins.child_frame_id  = "bolide_CoG";
    tf_ins.transform.translation.x = ins.x;
    tf_ins.transform.translation.y = ins.y;
    tf_ins.transform.translation.z = 0.0;
    tf2::Quaternion q2; q2.setRPY(0, 0, ins.yaw);
    tf_ins.transform.rotation.x = q2.x();
    tf_ins.transform.rotation.y = q2.y();
    tf_ins.transform.rotation.z = q2.z();
    tf_ins.transform.rotation.w = q2.w();
    tf_br_.sendTransform(tf_ins);
}



void Simulation_lem_ros_node::publish_cones_gt_markers_()
{
    visualization_msgs::MarkerArray arr;

    if (!pub_markers_cones_gt_) {
        std::cerr << "[PUB][ERROR] pub_markers_cones_gt_ invalid; NOT publishing GT markers." << std::endl;
        ROS_ERROR("pub_markers_cones_gt_ invalid; not publishing GT markers");
        return;
    }

    // (opcjonalnie) wyczyść poprzednie markery od tego publishera
    {
        visualization_msgs::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = ros::Time::now();
        del.action = visualization_msgs::Marker::DELETEALL;
        arr.markers.push_back(del);
    }

    // lifetime = 0 → wieczny
    const ros::Duration kForever(0.0);

    int id = 300;
    for (const auto& c : track_global_.cones)
    {
        // kolor wg klasy (yellow/blue/orange/…)
        std_msgs::ColorRGBA col = color_from_class_gt(c.color, 0.95f);

        // bazowy “stożek” jako cylinder; funkcja ustawia ns="cones",
        // za chwilę nadpiszemy na "cones_gt" i frame na "map"
        visualization_msgs::Marker m = make_cone_marker(
            id++, /*frame*/ "map", c.x, c.y, c.z, col, kForever
        );

        // doprecyzowanie nagłówka/namespacu dla GT
        m.header.frame_id = "map";
        m.header.stamp    = ros::Time::now();
        m.ns = "cones_gt";                  // osobna przestrzeń nazw dla GT
        m.action = visualization_msgs::Marker::ADD;

        // (opcjonalnie) możesz różnić GT od wizji np. większą przezroczystością:
        // m.color.a = 0.7f;

        arr.markers.push_back(std::move(m));
    }

    pub_markers_cones_gt_.publish(arr);
}

void Simulation_lem_ros_node::pub_full_state_()
{
    dv_interfaces::full_state msg;
    Log_Info_full info = log_info_full(
        state_,
        Input(torque_cmd_fl_, torque_cmd_fr_, torque_cmd_rl_, torque_cmd_rr_, steer_command_to_maxon_),
        P_,
        step_number_
    );

    msg.time = info.time;
    msg.step_number = step_number_;

    msg.x = info.x;
    msg.y = info.y;
    msg.yaw = info.yaw;
    msg.yaw_rate = info.yaw_rate;
    msg.vx = info.vx;

    msg.vy = info.vy;
    msg.ax = info.ax;
    msg.ay = info.ay;

    // ==========================================================
    // 4WD TORQUES (NA 4 KOŁA)
    // ==========================================================
    msg.torque_fl = info.torque_fl;
    msg.torque_fr = info.torque_fr;
    msg.torque_rl = info.torque_rl;
    msg.torque_rr = info.torque_rr;


    // ==========================================================
    // OMEGI 4WD
    // ==========================================================
    msg.omega_fl = info.omega_fl;
    msg.omega_fr = info.omega_fr;
    msg.omega_rl = info.omega_rl;
    msg.omega_rr = info.omega_rr;

    msg.rack_angle = info.rack_angle;
    msg.delta_left = info.delta_left;
    msg.delta_rigth = info.delta_rigth;
    msg.rack_angle_request = info.rack_angle_request;

    msg.fx_fl = info.fx_fl;
    msg.fx_fr = info.fx_fr;
    msg.fx_rl = info.fx_rl;
    msg.fx_rr = info.fx_rr;

    msg.fy_fl = info.fy_fl;
    msg.fy_fr = info.fy_fr;
    msg.fy_rl = info.fy_rl;
    msg.fy_rr = info.fy_rr;

    msg.fz_fl = info.fz_fl;
    msg.fz_fr = info.fz_fr;
    msg.fz_rl = info.fz_rl;
    msg.fz_rr = info.fz_rr;

    msg.slip_angle_fl = info.slip_angle_fl;
    msg.slip_angle_fr = info.slip_angle_fr;
    msg.slip_angle_rl = info.slip_angle_rl;
    msg.slip_angle_rr = info.slip_angle_rr;
    msg.slip_angle_body = info.slip_angle_body;

    msg.kappa_fl = info.kappa_fl;
    msg.kappa_fr = info.kappa_fr;
    msg.kappa_rl = info.kappa_rl;
    msg.kappa_rr = info.kappa_rr;

    msg.total_drag = info.total_drag;
    msg.total_downforce = info.total_downforce;
    msg.Power_total = info.Power_total;

    msg.step_dt = P_.get("simulation_time_step");

    // ==========================================================
    // TOP SLIP RATIO: 4WD -> max abs kappa z 4 kół
    // ==========================================================

    if( state_.vx > 1.0 ) {
        const double kappa_max_abs_signed = [&]() -> double
        {
            double best = msg.kappa_fl;
            if (std::abs(msg.kappa_fr) > std::abs(best)) best = msg.kappa_fr;
            if (std::abs(msg.kappa_rl) > std::abs(best)) best = msg.kappa_rl;
            if (std::abs(msg.kappa_rr) > std::abs(best)) best = msg.kappa_rr;
            return best;
        }();

        const double slip_angle_wheel_max_abs_signed = [&]() -> double
        {
            double best = msg.slip_angle_fl;
            if (std::abs(msg.slip_angle_fr) > std::abs(best)) best = msg.slip_angle_fr;
            if (std::abs(msg.slip_angle_rl) > std::abs(best)) best = msg.slip_angle_rl;
            if (std::abs(msg.slip_angle_rr) > std::abs(best)) best = msg.slip_angle_rr;
            return best;
        }();

        // ==========================================================
        // METRICS: slip ratio / slip angle uwzględniają WSZYSTKIE 4 KOŁA
        auto clamp_abs = [](double v, double max_abs) -> double {
            return std::min(std::abs(v), max_abs);
        };
        
        const double dt = msg.step_dt;
        
        // ---- slip ratio (kappa) ----
        const double kappa_thr = P_.get("kappa_slip_threshold");
        const double kappa_cap = 1.0; // clamp |kappa| to 1
        
        slip_ratio_metric_ += std::max(0.0, clamp_abs(msg.kappa_fl, kappa_cap) - kappa_thr) * dt;
        slip_ratio_metric_ += std::max(0.0, clamp_abs(msg.kappa_fr, kappa_cap) - kappa_thr) * dt;
        slip_ratio_metric_ += std::max(0.0, clamp_abs(msg.kappa_rl, kappa_cap) - kappa_thr) * dt;
        slip_ratio_metric_ += std::max(0.0, clamp_abs(msg.kappa_rr, kappa_cap) - kappa_thr) * dt;
        
        // ---- slip angles ----
        const double slip_ang_thr = P_.get("slip_angle_slip_threshold");
        const double slip_ang_cap = 90.0; 
        slip_angle_metric_ += std::max(0.0, clamp_abs(msg.slip_angle_fl, slip_ang_cap) - slip_ang_thr) * dt;
        slip_angle_metric_ += std::max(0.0, clamp_abs(msg.slip_angle_fr, slip_ang_cap) - slip_ang_thr) * dt;
        slip_angle_metric_ += std::max(0.0, clamp_abs(msg.slip_angle_rl, slip_ang_cap) - slip_ang_thr) * dt;
        slip_angle_metric_ += std::max(0.0, clamp_abs(msg.slip_angle_rr, slip_ang_cap) - slip_ang_thr) * dt;
        
        // ---- body slip angle ----
        const double slip_body_thr = P_.get("slip_body_slip_threshold");
        slip_angle_body_metric_ += std::max(0.0, clamp_abs(msg.slip_angle_body, slip_ang_cap) - slip_body_thr) * dt;

        update_top_abs(ten_biggest_slip_ratio_, kappa_max_abs_signed, 10);
        update_top_abs(ten_biggest_beta_angle_, slip_angle_wheel_max_abs_signed, 10);

        const double beta_thresh = 9.0;
        if (std::abs(msg.slip_angle_body) > beta_thresh) {
            time_beta_over_9_ += msg.step_dt;
        }
    }

    pub_log_full_.publish(msg);



    // ==========================================================
    // [NOWE] RYSOWANIE KROPLI G-G (MARKER)
    // ==========================================================
    visualization_msgs::Marker gg_sphere;
    gg_sphere.header.frame_id = "gg_dashboard"; // 
    gg_sphere.header.stamp = ros::Time::now();
    gg_sphere.ns = "gg_current_accel";
    gg_sphere.id = 1; // Musi być inny niż ID obwiedni!
    gg_sphere.type = visualization_msgs::Marker::SPHERE;
    gg_sphere.action = visualization_msgs::Marker::ADD;

    // Współrzędne na wykresie:
    // Oś X wykresu = Przyspieszenie boczne (ay)
    // Oś Y wykresu = Przyspieszenie wzdłużne (ax)
    gg_sphere.pose.position.x = info.ay; 
    gg_sphere.pose.position.y = info.ax;
    gg_sphere.pose.position.z = 0.0;
    
    // Brak rotacji
    gg_sphere.pose.orientation.w = 1.0; 

    // Rozmiar "kropli" (np. 15 cm średnicy na wykresie)
    gg_sphere.scale.x = 1.15;
    gg_sphere.scale.y = 1.15;
    gg_sphere.scale.z = 1.15; // Byłaby to kula, ale patrzymy z góry

    // Kolor - np. jaskrawy czerwony, żeby odcinał się od szarej obwiedni
    gg_sphere.color.r = 1.0;
    gg_sphere.color.g = 0.0;
    gg_sphere.color.b = 0.0;
    gg_sphere.color.a = 1.0; 

    pub_gg_sphere_marker_.publish(gg_sphere);
}
 
void Simulation_lem_ros_node::publish_bolid_marker_()
{
    if (!pub_marker_bolid_) {
        ROS_ERROR("pub_marker_bolid_ invalid; not publishing bolid marker");
        return;
    }

    visualization_msgs::Marker car;
    car.header.frame_id = "bolide_true";        // auto porusza się z TF pojazdu
    car.header.stamp    = ros::Time::now();
    car.ns   = "bolide";
    car.id   = 0;
    car.type = visualization_msgs::Marker::CUBE;
    car.action = visualization_msgs::Marker::ADD;

    // --- rozmiar pojazdu FS (około) ---
    car.scale.x = 2.8;    // długość (m)
    car.scale.y = 1.5;    // szerokość (m)
    car.scale.z = 1.5;    // wysokość (m)

    // --- pozycja ---
    car.pose.position.x = 0.0;      // środek ciężkości = TF origin
    car.pose.position.y = 0.0;
    car.pose.position.z = 0.75;     // połowa wysokości, żeby stał na ziemi
    car.pose.orientation.w = 1.0;

    car.color.r = 0.0f;
    car.color.g = 0.9f;
    car.color.b = 0.2f;
    car.color.a = 1.0f;

    // --- lifetime krótkie, auto się odświeża ---
    car.lifetime = ros::Duration(0.1);

    pub_marker_bolid_.publish(car);
}



void Simulation_lem_ros_node::log_metric_of_ride_data_()
{
    // Jeśli nie ma ścieżki, to nic nie zapisuję
    if (metrics_log_file_path_.empty()) return;

    std::ofstream f(metrics_log_file_path_, std::ios::out);
    if (!f.is_open())
    {
        ROS_WARN_STREAM("[METRICS] Cannot open metrics file: " << metrics_log_file_path_);
        return;
    }

    const double dt = P_.get("simulation_time_step");
    const double total_time = static_cast<double>(step_number_) * dt;

    // policz procent czasu TC aktywne dopiero na końcu
    if (total_time > 1e-9)
        percetage_of_time_tc_active_ = 100.0 * time_tc_active_ / total_time;
    else
        percetage_of_time_tc_active_ = 0.0;

    if (total_time > 1e-9)
        percetage_of_time_beta_over_9_ = 100.0 * time_beta_over_9_ / total_time;
    else
        percetage_of_time_beta_over_9_ = 0.0;

    // helper do wektorów -> "a;b;c;d"
    auto join_vec = [](const std::vector<double>& v) -> std::string
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < v.size(); ++i)
        {
            if (i) oss << ";";
            oss << v[i];
        }
        return oss.str();
    };

    // CSV: proste "metric,value" - łatwe do parsowania i czytania
    f << "metric,value\n";

    f << "total_time_s," << total_time << "\n";
    f << "ey_avg_m,"      << ey_avg_   << "\n";
    f << "epsi_avg_rad,"  << epsi_avg_ << "\n";
    f << "vs_avg_mps,"    << vs_avg_   << "\n";

    f << "time_tc_active_s," << time_tc_active_ << "\n";
    f << "tc_active_percent," << percetage_of_time_tc_active_ << "\n";
    f << "time_beta_over_9deg_s," << time_beta_over_9_ << "\n";
    f << "beta_over_9deg_percent," << percetage_of_time_beta_over_9_ << "\n";

    // Top 10 listy jako jedna komórka (bezpieczne i wygodne)
    f << "ten_biggest_slip_ratio," << "\"" << join_vec(ten_biggest_slip_ratio_) << "\"\n";
    f << "ten_biggest_beta_angle," << "\"" << join_vec(ten_biggest_beta_angle_) << "\"\n";
    f << "ten_biggest_ey,"         << "\"" << join_vec(ten_biggest_ey_) << "\"\n";
    f << "ten_biggest_epsi,"       << "\"" << join_vec(ten_biggest_epsi_) << "\"\n";
    f << "soft_track_violations_count," << soft_track_violation_count_/total_time << "\n";
f << "medium_track_violations_count," << medium_track_violation_count_/total_time << "\n";
f << "high_track_violations_count," << high_track_violation_count_/total_time << "\n";
    f << "slip_ratio_metric," << slip_ratio_metric_/total_time << "\n";
    f << "slip_angle_metric," << slip_angle_metric_/total_time << "\n";
    f << "slip_angle_body_metric," << slip_angle_body_metric_/total_time << "\n";

    // Informacje o crashu
    f << "crashed," << (crashed_ ? 1 : 0) << "\n";
    f << "crash_time_s," << crash_time_s_ << "\n";
    f << "crash_reason," << "\"" << crash_reason_ << "\"\n";

    f.flush();
    f.close();

    ROS_WARN_STREAM("[METRICS] Saved ride metrics to: " << metrics_log_file_path_);
}
void Simulation_lem_ros_node::mpc_debug_callback_(const dv_interfaces::MPCDebug::ConstPtr& msg)
{
    // ======================================================
    // METRICS (ey, epsi, vs): liczone z rzutu na splajn (jeśli dostępny)
    // fallback: wartości z msg
    // ======================================================
    double ey   = msg->ey_current;
    double epsi = msg->epsi_current;
    double vs   = msg->v_s_avg;

    if (center_line_spline_.valid()) {
        v2_control::Vec2 q(state_.x, state_.y);
        const double s_proj = center_line_spline_.projectToSpline(q);
        const v2_control::Vec2 p = center_line_spline_.eval(s_proj);

        const double dx = state_.x - p.x;
        const double dy = state_.y - p.y;
        const double dist = std::hypot(dx, dy);

        const double path_yaw = center_line_spline_.getYaw(s_proj);
        const double tx = std::cos(path_yaw);
        const double ty = std::sin(path_yaw);

        // cross(tangent, delta_pos) = tx*dy - ty*dx -> znak ey
        const double c = tx * dy - ty * dx;
        ey = (c >= 0.0) ? dist : -dist;

        epsi = state_.yaw - path_yaw;
        while (epsi >  M_PI) epsi -= 2.0 * M_PI;
        while (epsi < -M_PI) epsi += 2.0 * M_PI;

        const double kappa = center_line_spline_.getCurvature(s_proj);
        const double denom = 1.0 - kappa * ey;
        const double denom_safe = (std::abs(denom) < 1e-6) ? ((denom >= 0.0) ? 1e-6 : -1e-6) : denom;

        // tangential velocity along path
        vs = (state_.vx * std::cos(epsi) - state_.vy * std::sin(epsi)) / denom_safe;
    }

    // ======================================================
    // RUNNING METRICS
    // ======================================================
    static double sum_sq_ey   = 0.0;
    static double sum_sq_epsi = 0.0;
    static double sum_vs      = 0.0;
    static long long count_metrics = 0;

    sum_sq_ey   += ey   * ey;
    sum_sq_epsi += epsi * epsi;
    sum_vs      += vs;
    count_metrics = std::max<long long>(1, count_metrics + 1);

    ey_avg_   = std::sqrt(sum_sq_ey   / static_cast<double>(count_metrics));
    epsi_avg_ = std::sqrt(sum_sq_epsi / static_cast<double>(count_metrics));
    vs_avg_   = sum_vs / static_cast<double>(count_metrics);

    update_top_abs(ten_biggest_ey_,   std::abs(ey),   10);
    update_top_abs(ten_biggest_epsi_, std::abs(epsi), 10);

    if (crashed_) return;

    // ======================================================
    // OPTIONAL: solver fail
    // ======================================================
    if (msg->solver_failed)
    {
        // jeśli chcesz jednak crashować na solverze, odkomentuj:
        // mark_crash_("mpc_solver_failed=1");
        return;
    }

    // ======================================================
    // YAW-RATE CRASH
    // ======================================================
    if (center_line_spline_.valid())
    {
        const double s_proj = center_line_spline_.projectToSpline(v2_control::Vec2(state_.x, state_.y));
        const double kappa  = center_line_spline_.getCurvature(s_proj);

        const double yr_mpc  = state_.yaw_rate;
        const double yr_curv = state_.vx * kappa;

        const double factor  = P_.get("max_yaw_rate_factor_violation");
        const double eps_num = 1e-6;

        if (std::abs(yr_mpc) > factor * (std::abs(yr_curv) + eps_num) &&
            std::abs(yr_curv) > 1.25)
        {
            std::ostringstream oss;
            oss << "yaw_rate_violation: |yr_mpc|=" << std::abs(yr_mpc)
                << " > " << factor << "*(|yr_curv|+eps)=" << factor * (std::abs(yr_curv) + eps_num);
            mark_crash_(oss.str());
            return;
        }
    }

    // ======================================================
    // TRACK VIOLATION CLASSIFICATION
    // crash TYLKO dla critical
    // ======================================================
    {
        const double L = P_.get("car_length");
        const double W = P_.get("car_width");

        const double track_half_width = 0.5 * P_.get("track_width");

        const double thr_soft     = P_.get("max_track_violation_soft");
        const double thr_medium   = P_.get("max_track_violation_medium");
        const double thr_high     = P_.get("max_track_violation_high");
        const double thr_critical = critical_crash_multiplier_ * P_.get("max_track_violation_critical_crash");

        // zajętość boczna samochodu w lokalnym przekroju toru
        const double req = 0.5 * L * std::sin(std::abs(epsi))
                         + 0.5 * W * std::abs(std::cos(epsi));

        // ile "wystaje" ponad pół-szerokość toru
        const double left_excess  = std::max(0.0, ey  + req - track_half_width);
        const double right_excess = std::max(0.0, -ey + req - track_half_width);
        const double track_excess = std::max(left_excess, right_excess);

        if (track_excess > thr_soft) {
            soft_track_violation_count_++;
        }
        if (track_excess > thr_medium) {
            medium_track_violation_count_++;
        }
        if (track_excess > thr_high) {
            high_track_violation_count_++;
        }
        if (track_excess > thr_critical) {
            critical_track_violation_count_++;

            std::ostringstream oss;
            oss << "critical_track_violation: excess=" << track_excess
                << " > thr_critical=" << thr_critical
                << " | ey=" << ey
                << " | epsi=" << epsi
                << " | req=" << req
                << " | track_half_width=" << track_half_width
                << " | left_excess=" << left_excess
                << " | right_excess=" << right_excess;

            mark_crash_(oss.str());
            return;
        }
    }

    // ======================================================
    // AVG-BASED CRASHES
    // ======================================================
    const double t = step_number_ * P_.get("simulation_time_step");

    if (ey_avg_ > P_.get("ey_avg_crash_threshold") && t > 10.0)
    {
        std::ostringstream oss;
        oss << "ey_avg_violation: ey_avg=" << ey_avg_
            << " > thr=" << P_.get("ey_avg_crash_threshold")
            << " | ey_current=" << ey
            << " | t=" << t;
        mark_crash_(oss.str());
        return;
    }

    if (vs_avg_ < P_.get("vs_avg_crash_threshold") && t > 10.0)
    {
        std::ostringstream oss;
        oss << "vs_avg_violation: vs_avg=" << vs_avg_
            << " < thr=" << P_.get("vs_avg_crash_threshold")
            << " | t=" << t;
        mark_crash_(oss.str());
        return;
    }
}

Torque_allocation Simulation_lem_ros_node::allocate_torque_optimaly(double fx_target , double mz_target ){

    // =========================================================
    // PARAMETERS FOR WEIGHT DISTRIBUTION CALCULATION
    const double m = P_.get("m");
    const double g = P_.get("g");
    const double w = P_.get("w");
    const double a = P_.get("a");
    const double b = P_.get("b");
    const double t_front = P_.get("t_front");
    const double t_rear  = P_.get("t_rear");
    const double h = P_.get("h");
    const double h_roll_f = P_.get("h1_roll");
    const double h_roll_r = P_.get("h2_roll");

    const double Kf = P_.get("K1");
    const double Kr = P_.get("K2");
    const double K_total = Kf + Kr;

    const double mf = m * a / w;
    const double mr = m * b / w;
    const double h_prim_f = h - h_roll_f;
    const double h_prim_r = h - h_roll_r;
   
    double ax_messured = last_imu_.ax;
    double ay_messured = last_imu_.ay;
    double vx = state_.vx;
    double Fz_total = P_.get("m") * P_.get("g") + (P_.get("Cl1") + P_.get("Cl2")) * state_.vx * state_.vx;

    double N_fl = 0.5 * mf * g - 0.5 * m * ax_messured * h / w
                - ay_messured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx * vx;

    double N_fr = 0.5 * mf * g - 0.5 * m *ax_messured * h / w
                + ay_messured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx * vx;

    double N_rl = 0.5 * mr * g + 0.5 * m * ax_messured * h / w
                - ay_messured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx * vx;

    double N_rr = 0.5 * mr * g + 0.5 * m * ax_messured * h / w
                + ay_messured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx * vx;

    const double FZ_MIN = 50.0;

    N_fl = std::max(N_fl, FZ_MIN); // for numerical stability
    N_fr = std::max(N_fr, FZ_MIN); // for numerical stability
    N_rl = std::max(N_rl, FZ_MIN); // for numerical stability
    N_rr = std::max(N_rr, FZ_MIN); // for numerical stability

    // heurystic approach returned in case of solver failure/to test if better
    Torque_allocation u_heuristic;
    // distributing longitudinal force according to nor ;
    u_heuristic.torque_fr = fx_target * N_fr / Fz_total * P_.get("R");
    u_heuristic.torque_fl = fx_target * N_fl / Fz_total * P_.get("R");
    u_heuristic.torque_rl = fx_target * N_rl / Fz_total * P_.get("R");
    u_heuristic.torque_rr = fx_target * N_rr / Fz_total * P_.get("R");
    // the distribution of mz according to the normal forces on axles
    double mz_front = mz_target * (N_fl + N_fr) / Fz_total;
    double mz_rear  = mz_target * (N_rr + N_rl) / Fz_total;
    double d_f_front = mz_front / t_front;
    double d_f_rear  = mz_rear / t_rear;
    u_heuristic.torque_fl += -d_f_front * P_.get("R");
    u_heuristic.torque_fr += +d_f_front * P_.get("R");
    u_heuristic.torque_rl += -d_f_rear * P_.get("R"); ;
    u_heuristic.torque_rr += +d_f_rear * P_.get("R"); ;
    // Solving QP for allocationg torque to wheels to follow fx and mz targets with taking into account current weight distribution 
    // using cholesky decomposition method
    // u := [F_fl, F_fr, F_rl, F_rr]^T  (forces on wheels)
    // u_hat := [Fx_target, Mz_target]^T
    // J(u) = (G u - u_hat)^T Q (G u - u_hat) + u^T R u
    //
    // Optimality condition:
    // (G^T Q G + R) u = G^T Q u_hat
    // A u = c, where A := G^T Q G + R (SPD), c := G^T Q u_hat
    //
    // I solve with Cholesky: A = L L^T

    //Cost params of QP

    
    const double lambda = P_.get("torque_allocation_lambda_factor");
    const double Fx_ref = P_.get("torque_allocation_fx_ref");
    const double Mz_ref = P_.get("torque_allocation_mz_ref");
    Eigen::Matrix<double,2,4> G;
    G << 1.0, 1.0, 1.0, 1.0,
        -0.5*t_front, +0.5*t_front, -0.5*t_rear, +0.5*t_rear;

    // =========================
    // 2) Build Q (2x2), diag weights for tracking errors
    //    Typical: qFx = 1/Fx_ref^2, qMz = 1/Mz_ref^2
    // =========================
    const double qFx = 1.0 / (Fx_ref * Fx_ref);
    const double qMz = 1.0 / (Mz_ref * Mz_ref);

    Eigen::Matrix2d Q = Eigen::Matrix2d::Zero();
    Q(0,0) = qFx;
    Q(1,1) = qMz;

    Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
    const double r_fl = Fz_total/N_fl *qFx ;
    const double r_fr = Fz_total/N_fr *qFx ;
    const double r_rl = Fz_total/N_rl *qFx ;
    const double r_rr = Fz_total/N_rr *qFx ;

    R(0,0) = lambda*r_fl;
    R(1,1) = lambda*r_fr;
    R(2,2) = lambda*r_rl;
    R(3,3) = lambda*r_rr;
    // =========================
    // 4) Build u_hat (2x1)
    // =========================
    Eigen::Vector2d u_hat;
    u_hat << fx_target, mz_target;


       // =========================
    // 5) Build A and b:
    //    A = G^T Q G + R
    //    c = G^T Q u_hat
    // =========================
    Eigen::Matrix4d A = G.transpose() * Q * G + R;
    Eigen::Vector4d c = G.transpose() * Q * u_hat;

    // =========================
    // 6) Solve A u = c using Cholesky (LLT)
    // =========================
    Eigen::LLT<Eigen::Matrix4d> llt(A);
    if (llt.info() != Eigen::Success) {
        return u_heuristic;
    }
    
    Eigen::Vector4d u = llt.solve(c);
    if (llt.info() != Eigen::Success || !u.allFinite()) {
        return u_heuristic;
    }

    // std::cout << "Torque allocation solution: " << u.transpose() << std::endl;
    // // std::cout << "Heuristic solution: " << u_heuristic.torque_fl / P_.get("R") << ", "
    // //                                     << u_heuristic.torque_fr / P_.get("R") << ", "
    // //                                     << u_heuristic.torque_rl / P_.get("R") << ", "
    // //                                     << u_heuristic.torque_rr / P_.get("R") << std::endl;
    // std::cout << "residual: " << (G * u - u_hat).transpose() << std::endl;
    // std::cout << "Weights (N): " << N_fl << ", " << N_fr << ", " << N_rl << ", " << N_rr << std::endl;

    Torque_allocation torque_allocation_result;
    torque_allocation_result.torque_fl = u(0) * P_.get("R");
    torque_allocation_result.torque_fr = u(1) * P_.get("R");
    torque_allocation_result.torque_rl = u(2) * P_.get("R");
    torque_allocation_result.torque_rr = u(3) * P_.get("R");

    torque_allocation_result.torque_fl = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), torque_allocation_result.torque_fl));
    torque_allocation_result.torque_fr = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), torque_allocation_result.torque_fr  ));
    torque_allocation_result.torque_rl = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), torque_allocation_result.torque_rl));
    torque_allocation_result.torque_rr = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), torque_allocation_result.torque_rr));
    
    return torque_allocation_result ; // wheel torques in Nm
    
};

Torque_allocation Simulation_lem_ros_node::allocate_torque_heuristically(double  fx_target , double mz_target ){

     // =========================================================
    // PARAMETERS FOR WEIGHT DISTRIBUTION CALCULATION
    const double m = P_.get("m");
    const double g = P_.get("g");
    const double w = P_.get("w");
    const double a = P_.get("a");
    const double b = P_.get("b");
    const double t_front = P_.get("t_front");
    const double t_rear  = P_.get("t_rear");
    const double h = P_.get("h");
    const double h_roll_f = P_.get("h1_roll");
    const double h_roll_r = P_.get("h2_roll");

    const double Kf = P_.get("K1");
    const double Kr = P_.get("K2");
    const double K_total = Kf + Kr;

    const double mf = m * a / w;
    const double mr = m * b / w;
    const double h_prim_f = h - h_roll_f;
    const double h_prim_r = h - h_roll_r;
   
    double ax_messured = last_imu_.ax;
    double ay_messured = last_imu_.ay;
    double vx = state_.vx;
    double Fz_total = P_.get("m") * P_.get("g") + (P_.get("Cl1") + P_.get("Cl2")) * state_.vx * state_.vx;

    double N_fl = 0.5 * mf * g - 0.5 * m * ax_messured * h / w
                - ay_messured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx * vx;

    double N_fr = 0.5 * mf * g - 0.5 * m *ax_messured * h / w
                + ay_messured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx * vx;

    double N_rl = 0.5 * mr * g + 0.5 * m * ax_messured * h / w
                - ay_messured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx * vx;

    double N_rr = 0.5 * mr * g + 0.5 * m * ax_messured * h / w
                + ay_messured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx * vx;

    const double FZ_MIN = 100.0;
    double sumN = N_fl + N_fr + N_rl + N_rr;

    N_fl = std::max(N_fl, FZ_MIN); // for numerical stability
    N_fr = std::max(N_fr, FZ_MIN); // for numerical stability
    N_rl = std::max(N_rl, FZ_MIN); // for numerical stability
    N_rr = std::max(N_rr, FZ_MIN); // for numerical stability
    

    // heurystic approach returned in case of solver failure/to test if better
    Torque_allocation u_heuristic;
    // distributing longitudinal force according to nor ;
    u_heuristic.torque_fr = fx_target * N_fr / Fz_total * P_.get("R");
    u_heuristic.torque_fl = fx_target * N_fl / Fz_total * P_.get("R");
    u_heuristic.torque_rl = fx_target * N_rl / Fz_total * P_.get("R");
    u_heuristic.torque_rr = fx_target * N_rr / Fz_total * P_.get("R");
    // the distribution of mz according to the normal forces on axles
    double mz_front = mz_target * (N_fl + N_fr) / Fz_total;
    double mz_rear  = mz_target * (N_rr + N_rl) / Fz_total;
    double d_f_front = mz_front / t_front;
    double d_f_rear  = mz_rear / t_rear;
    u_heuristic.torque_fl += -d_f_front * P_.get("R");
    u_heuristic.torque_fr += +d_f_front * P_.get("R");
    u_heuristic.torque_rl += -d_f_rear * P_.get("R"); ;
    u_heuristic.torque_rr += +d_f_rear * P_.get("R"); ;

    u_heuristic.torque_fl = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), u_heuristic.torque_fl));
    u_heuristic.torque_fr = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), u_heuristic.torque_fr));
    u_heuristic.torque_rl = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), u_heuristic.torque_rl));
    u_heuristic.torque_rr = std::max(-P_.get("max_torque"), std::min(P_.get("max_torque"), u_heuristic.torque_rr));

    return u_heuristic; // wheel torques in Nm
}
void Simulation_lem_ros_node::apply_traction_control_4wd_(double Ts)
{
    if (!lov_level_control_on) return;

    // ============================================================
    // 0. PARAMETRY POJAZDU I OBLICZANIE NACISKÓW (Fz)
    // ============================================================
    const double m = P_.get("m");
    const double g = P_.get("g");
    const double w = P_.get("w");
    const double a = P_.get("a");
    const double b = P_.get("b");
    const double t_front = P_.get("t_front");
    const double t_rear  = P_.get("t_rear");
    const double h = P_.get("h");
    const double h_roll_f = P_.get("h1_roll");
    const double h_roll_r = P_.get("h2_roll");
    const double Kf = P_.get("K1");
    const double Kr = P_.get("K2");
    const double K_total = Kf + Kr;

    const double mf = m * a / w;
    const double mr = m * b / w;
    const double h_prim_f = h - h_roll_f;
    const double h_prim_r = h - h_roll_r;

    const double ax_measured = last_imu_.ax;
    const double ay_measured = last_imu_.ay;
    const double vx_state    = state_.vx;

    // Ja szacuję naciski dynamiczne
    double N_fl = 0.5 * mf * g - 0.5 * m * ax_measured * h / w
                - ay_measured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx_state * vx_state;

    double N_fr = 0.5 * mf * g - 0.5 * m * ax_measured * h / w
                + ay_measured / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl1") * vx_state * vx_state;

    double N_rl = 0.5 * mr * g + 0.5 * m * ax_measured * h / w
                - ay_measured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx_state * vx_state;

    double N_rr = 0.5 * mr * g + 0.5 * m * ax_measured * h / w
                + ay_measured / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P_.get("Cl2") * vx_state * vx_state;

    const double FZ_MIN = 100.0;
    N_fl = std::max(N_fl, FZ_MIN);
    N_fr = std::max(N_fr, FZ_MIN);
    N_rl = std::max(N_rl, FZ_MIN);
    N_rr = std::max(N_rr, FZ_MIN);

    // ============================================================
    // 1. STATYSTYKI I PREDYKCJA PRĘDKOŚCI (OFFS / RAW / CURR)
    // ============================================================
    static double   total_ax_sum = 0.0;
    static uint64_t total_ticks  = 0;

    total_ax_sum += ax_measured;
    total_ticks++;

    const double avg_ax = (total_ticks > 0) ? (total_ax_sum / total_ticks) : 0.0;

    const double raw_gps_speed = static_cast<double>(last_input_read_by_dv_board.current_speed);

    static double last_gps_sample    = 0.0;
    static double predicted_offset   = 0.0;

    if (std::abs(raw_gps_speed - last_gps_sample) > 1e-7) {
        last_gps_sample  = raw_gps_speed;
        predicted_offset = 0.0; // Ja synchronizuję offset z nową próbką GPS
    } else {
        // Ja całkuję ax, żeby wypełnić luki 10Hz GPS-a
        predicted_offset += std::max(0.0, ax_measured) * Ts;
    }

    const double speed_current = raw_gps_speed + predicted_offset;

    static int log_counter = 0;
    const bool should_log = (log_counter++ % 10 == 0); // 100Hz debugowania

    // if (should_log) {
    //     std::cout << "\n========================= TC 100Hz DEBUG =========================\n";
    //     std::cout << "  AVG AX: " << std::fixed << std::setprecision(2) << avg_ax << " m/s^2\n";
    //     std::cout << "  SPEED:  Raw(GPS): " << std::setw(5) << raw_gps_speed
    //               << " | Offs: " << std::setw(5) << predicted_offset
    //               << " | Curr: " << std::setw(5) << speed_current << " m/s\n";
    //     std::cout << "------------------------------------------------------------------\n";
    //     std::cout << std::left << std::setw(5) << "Wh"
    //               << std::setw(7) << "Err"
    //               << std::setw(8) << "P-term"
    //               << std::setw(8) << "I-term"
    //               << std::setw(8) << "D-term"
    //               << std::setw(8) << "Lim"
    //               << "Final\n";
    // }

    // ============================================================
    // PID enable gate (poniżej 2 m/s: tylko feedforward, PID OFF)
    // ============================================================
    const double PID_ENABLE_SPEED = 2.0;
    const bool pid_enabled =  (speed_current >= PID_ENABLE_SPEED);

    // ============================================================
    // 2. LAMBDA: DRIVE TC + BRAKE TC (ABS) + gating < 2 m/s
    // ============================================================
    auto apply_wheel = [&](const std::string& name,
                           double w_i,            // [m/s] prędkość obwodowa koła
                           double& T_i,           // [Nm] komenda momentu ( +drive, -brake/regen )
                           double N_i,            // [N] nacisk koła
                           PIDController& pid_drive,
                           PIDController& pid_brake)
    {
        // --- FEEDFORWARD: limit momentu na podstawie Fz ---
        const double mu_peak = 1.4;
        const double Rw = 0.195;
        const double T_max_tire = N_i * mu_peak * Rw;

        // Ja zawsze ograniczam wejściowy moment do możliwości opony (symetrycznie)
        const double T_cmd_limited = std::clamp(T_i, -T_max_tire, +T_max_tire);

        // ------------------------------------------------------------
        // GATE: poniżej 2 m/s -> tylko feedforward, PID-y WYŁĄCZONE
        // ------------------------------------------------------------
        if (!pid_enabled)
        {
            T_i = T_cmd_limited;

            // Ja robię leak obu regulatorów
            pid_drive.update(0.0, Ts, false);
            pid_brake.update(0.0, Ts, false);

            // if (should_log) {
            //     std::cout << std::left << std::setw(5) << name
            //               << std::setw(7) << std::setprecision(2) << 0.0
            //               << std::setw(8) << 0.0
            //               << std::setw(8) << 0.0
            //               << std::setw(8) << 0.0
            //               << std::setw(8) << (int)T_max_tire
            //               << std::setprecision(1) << T_i << "Nm\n";
            // }
            return;
        }

        // ------------------------------------------------------------
        // Powyżej progu: działam zależnie od znaku momentu
        // ------------------------------------------------------------

        // ===== DRIVE (TC) =====
        if (T_i > 0.0)
        {
            // target prędkości koła (slip dodatni)
            const double target_w = speed_current * (1.0 + P_.get("target_slip_drive"));

            // overspeed tylko gdy koło za szybkie
            const double err_raw   = w_i - target_w;
            double err_drive = std::max(0.0, err_raw);
              // Ja pozwalam na ujemny błąd (koło wolniejsze niż target), żeby PID mógł reagować w obu kierunkach
            err_drive /= speed_current; // Ja normalizuję błąd do względnego slipu

            // PID drive aktywny, PID brake w leak
            pid_drive.update(err_drive, Ts, true,false);
          

            pid_brake.update(0.0, Ts, false);

            // Ja biorę tylko redukcję (nie pozwalam na ujemną redukcję)
            const double reduction = std::max(0.0, pid_drive.get_output());

            // TC może zejść do lekkiego regen (anty-lag / stabilizacja)
            const double regen_limit = -0.0;

            T_i = std::max(regen_limit, T_cmd_limited - reduction);

            // if (should_log) {
            //     std::cout << std::left << std::setw(5) << name
            //               << std::setw(7) << std::setprecision(2) << err_drive
            //               << std::setw(8) << pid_drive.get_P_term()
            //               << std::setw(8) << pid_drive.get_I_term_integrator()
            //               << std::setw(8) << pid_drive.get_D_term()
            //               << std::setw(8) << (int)T_max_tire
            //               << std::setprecision(1) << T_i << "Nm\n";
            // }
            return;
        }

        // ===== BRAKE (ABS / TC-brake) =====
        if (T_i < 0.0)
        {
            // target prędkości koła dla hamowania:
            // koło może być trochę wolniejsze niż auto, ale nie za dużo (żeby nie blokowało)
            const double target_w_brake = speed_current * (1.0 - P_.get("target_slip_brake"));

            // lock-up gdy koło ZA WOLNE: w_i < target
            const double err_raw   = target_w_brake - w_i;
            double err_brake = std::max(0.0, err_raw);
            err_brake /= speed_current; // Ja normalizuję błąd do względnego slipu

            pid_brake.update(err_brake, Ts, true);

            pid_drive.update(0.0, Ts, false);

            // Ja robię tylko "odpuszczanie hamulca": dodatnia korekta zbliża moment do 0
            const double release = std::max(0.0, pid_brake.get_output());

            // ABS może tylko odpuszczać (i nigdy nie przejdzie w +)
            // T_cmd_limited jest ujemne; dodaję release (zmniejszam |T|)
            T_i = std::min(0.0, T_cmd_limited + release);

            // if (should_log) {
            //     std::cout << std::left << std::setw(5) << name
            //               << std::setw(7) << std::setprecision(2) << err_brake
            //               << std::setw(8) << pid_brake.get_P_term()
            //               << std::setw(8) << pid_brake.get_I_term_integrator()
            //               << std::setw(8) << pid_brake.get_D_term()
            //               << std::setw(8) << (int)T_max_tire
            //               << std::setprecision(1) << T_i << "Nm\n";
            // }
            return;
        }

        // ===== ZERO TORQUE =====
        pid_drive.update(0.0, Ts, false);
        pid_brake.update(0.0, Ts, false);

        if (should_log) {
            std::cout << std::left << std::setw(5) << name
                      << std::setw(7) << std::setprecision(2) << 0.0
                      << std::setw(8) << 0.0
                      << std::setw(8) << 0.0
                      << std::setw(8) << 0.0
                      << std::setw(8) << (int)T_max_tire
                      << std::setprecision(1) << T_i << "Nm\n";
        }
    };

    // ============================================================
    // 3. WYWOŁANIE DLA KAŻDEGO KOŁA
    // ============================================================
    apply_wheel("FL", wheel_speed_fl_, torque_cmd_fl_, N_fl, tc_drive_fl_, tc_brake_fl_);
    apply_wheel("FR", wheel_speed_fr_, torque_cmd_fr_, N_fr, tc_drive_fr_, tc_brake_fr_);
    apply_wheel("RL", wheel_speed_rl_, torque_cmd_rl_, N_rl, tc_drive_rl_, tc_brake_rl_);
    apply_wheel("RR", wheel_speed_rr_, torque_cmd_rr_, N_rr, tc_drive_rr_, tc_brake_rr_);

  //  if (should_log) std::cout << "==================================================================\n";

    const bool tc_active_now =
        tc_drive_fl_.is_active() || tc_drive_fr_.is_active() ||
        tc_drive_rl_.is_active() || tc_drive_rr_.is_active() ||
        tc_brake_fl_.is_active() || tc_brake_fr_.is_active() ||
        tc_brake_rl_.is_active() || tc_brake_rr_.is_active();

    if (tc_active_now) time_tc_active_ += Ts;
}

void Simulation_lem_ros_node::publish_rack_angle_()
{
    if (!is_due(step_number_, step_mcu_reading_, phase_mcu_reading_)) return;

    dv_interfaces::RackAngleSensor msg;
    msg.rack_angle = state_.rack_angle; 
    msg.rack_angle_velocity = state_.d_rack_angle ;
    msg.torque = (torque_cmd_fl_ + torque_cmd_fr_ + torque_cmd_rl_ + torque_cmd_rr_)/4.0/P_.get("max_torque"); // sumaryczny moment na kołach jako proxy momentu na racku
    pub_rack_angle_.publish(msg);
}


} // namespace lem_dynamics_sim_
