#pragma once
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace lem_dynamics_sim_ {

struct ParamBank {
  std::vector<std::string> names;
  std::unordered_map<std::string,int> idx;
  std::vector<double> values;

  int add(const std::string& name, double val){
    auto it = idx.find(name);
    if (it != idx.end()){
      values[it->second] = val;
      return it->second;
    }
    int k = (int)names.size();
    names.push_back(name);
    idx[name] = k;
    values.push_back(val);
    return k;
  }

  int i(const std::string& name) const {
    auto it = idx.find(name);
    if (it == idx.end()) throw std::runtime_error("ParamBank: missing key '"+name+"'");
    return it->second;
  }

  double get(const std::string& name) const { return values.at(i(name)); }
  void   set(const std::string& name, double v){ values.at(i(name)) = v; }
  size_t size() const { return values.size(); }
};

// --- Wymagane pole ---
inline double JgetReq(const nlohmann::json& J, const std::string& path){
  auto pos = path.find('.');
  if (pos == std::string::npos) {
    if (!J.contains(path))
      throw std::runtime_error("JSON: missing required key '"+path+"'");
    return J.at(path).get<double>();
  }
  std::string head = path.substr(0,pos);
  std::string tail = path.substr(pos+1);
  if (!J.contains(head))
    throw std::runtime_error("JSON: missing object '"+head+"'");
  return JgetReq(J.at(head), tail);
}

inline double JgetReqSafe(const nlohmann::json& J, const std::string& path) {
  try {
    return JgetReq(J, path);
  } catch (const std::exception& e) {
    std::cerr << "\n[JSON ERROR] while reading path: " << path
              << "\n  what(): " << e.what() << "\n" << std::endl;
    throw;
  }
}

inline double JgetOpt(const nlohmann::json& J, const std::string& path, double def){
  try { return JgetReq(J, path); } catch(...) { return def; }
}

// ============================================================
//  Budowa banku parametrów z pliku JSON
//  Uwaga: klucze ParamBank (np. "max_steer") są płaskie,
//  ale ich wartości są w JSON pod sekcjami vehicle/tire/...
// ============================================================
inline ParamBank build_param_bank(const nlohmann::json& J){
  ParamBank P;

  // --- Vehicle dynamics ---
  P.add("g",   JgetReqSafe(J,"vehicle.g"));
  P.add("m",   JgetReqSafe(J,"vehicle.m"));
  P.add("h",   JgetReqSafe(J,"vehicle.h"));
  P.add("w",   JgetReqSafe(J,"vehicle.w"));
  P.add("b",   JgetReqSafe(J,"vehicle.b"));
  P.add("h1_roll", JgetReqSafe(J,"vehicle.h1_roll"));
  P.add("h2_roll", JgetReqSafe(J,"vehicle.h2_roll"));

  const double a_val = JgetOpt(J, "vehicle.a", P.get("w") - P.get("b"));
  P.add("a", a_val);

  P.add("t_front",   JgetReqSafe(J,"vehicle.T_front"));
  P.add("t_rear",    JgetReqSafe(J,"vehicle.T_rear"));
  P.add("K1",        JgetReqSafe(J,"vehicle.K1"));
  P.add("K2",        JgetReqSafe(J,"vehicle.K2"));
  P.add("Cd",        JgetReqSafe(J,"vehicle.Cd"));
  P.add("Cl1",       JgetReqSafe(J,"vehicle.Cl1"));
  P.add("Cl2",       JgetReqSafe(J,"vehicle.Cl2"));
  P.add("Cr",        JgetReqSafe(J,"vehicle.Cr"));
  P.add("Iz",        JgetReqSafe(J,"vehicle.Iz"));
  P.add("R",         JgetReqSafe(J,"vehicle.R"));
  P.add("I_tire",    JgetReqSafe(J,"vehicle.I_tire"));
  P.add("angle_construction_front", JgetReqSafe(J,"vehicle.angle_construction_front"));
  P.add("angle_construction_rear",  JgetReqSafe(J,"vehicle.angle_construction_rear"));
  P.add("r_front",   JgetReqSafe(J,"vehicle.r_front"));
  P.add("r_rear",    JgetReqSafe(J,"vehicle.r_rear"));
  P.add("car_length" ,  JgetReqSafe(J,"vehicle.length"));
  P.add("car_width" ,   JgetReqSafe(J,"vehicle.width"));
// --- Tire parameters (MF6.1, no camber) ---

  // Longitudinal
  P.add("pCx1",     JgetReqSafe(J,"tire.pCx1"));

  P.add("pDx1",     JgetReqSafe(J,"tire.pDx1"));
  P.add("pDx2",     JgetReqSafe(J,"tire.pDx2"));

  P.add("pEx1",     JgetReqSafe(J,"tire.pEx1"));
  P.add("pEx2",     JgetReqSafe(J,"tire.pEx2"));
  P.add("pEx3",     JgetReqSafe(J,"tire.pEx3"));
  P.add("pEx4",     JgetReqSafe(J,"tire.pEx4"));

  P.add("pKx1",     JgetReqSafe(J,"tire.pKx1"));
  P.add("pKx2",     JgetReqSafe(J,"tire.pKx2"));
  P.add("pKx3",     JgetReqSafe(J,"tire.pKx3"));

  P.add("pHx1",     JgetReqSafe(J,"tire.pHx1"));
  P.add("pHx2",     JgetReqSafe(J,"tire.pHx2"));

  P.add("pVx1",     JgetReqSafe(J,"tire.pVx1"));
  P.add("pVx2",     JgetReqSafe(J,"tire.pVx2"));

  P.add("lambda_x", JgetReqSafe(J,"tire.lambda_x"));


  // Lateral
  P.add("pCy1",     JgetReqSafe(J,"tire.pCy1"));

  P.add("pDy1",     JgetReqSafe(J,"tire.pDy1"));
  P.add("pDy2",     JgetReqSafe(J,"tire.pDy2"));

  P.add("pEy1",     JgetReqSafe(J,"tire.pEy1"));
  P.add("pEy2",     JgetReqSafe(J,"tire.pEy2"));
  P.add("pEy3",     JgetReqSafe(J,"tire.pEy3"));

  P.add("pKy1",     JgetReqSafe(J,"tire.pKy1"));
  P.add("pKy2",     JgetReqSafe(J,"tire.pKy2"));
  P.add("pKy4",     JgetReqSafe(J,"tire.pKy4"));   // jeśli zostawiasz w kodzie (możesz nie używać, ale w configu jest)

  P.add("pHy1",     JgetReqSafe(J,"tire.pHy1"));
  P.add("pHy2",     JgetReqSafe(J,"tire.pHy2"));

  P.add("pVy1",     JgetReqSafe(J,"tire.pVy1"));
  P.add("pVy2",     JgetReqSafe(J,"tire.pVy2"));

  P.add("lambda_y", JgetReqSafe(J,"tire.lambda_y"));
  P.add("N0",       JgetReqSafe(J,"tire.N0"));
  P.add("epsilon",  JgetReqSafe(J,"tire.epsilon"));


  P.add("relax_time_slip_angle_first_guees", JgetReqSafe(J,"tire.relax_length_slip_angle_piorek"));
  P.add("relax_time_slip_ratio_first_guees", 0.4 * P.get("relax_time_slip_angle_first_guees"));
  P.add("cy", JgetReqSafe(J,"tire.cy_first_guees"));
  P.add("dy", 0.3 * P.get("cy") * P.get("relax_time_slip_angle_first_guees"));
  P.add("cx", JgetReqSafe(J,"tire.cx_first_guees"));
  P.add("dx", 0.3 * P.get("cx") * P.get("relax_time_slip_ratio_first_guees"));

  // --- Drivetrain ---
  P.add("P_max_drive",          JgetReqSafe(J,"drivetrain.P_max_drive"));
  P.add("P_min_recup",          JgetReqSafe(J,"drivetrain.P_min_recup"));
  P.add("drivetrain_timescale", JgetReqSafe(J,"drivetrain.drivetrain_timescale"));

  // Uwaga: wcześniej miałeś 0.95*... (sprawność). Jeśli to ma być w configu, przenieś do JSON.
  P.add("max_torque",           JgetReqSafe(J,"drivetrain.max_torque"));
  P.add("min_torque",           JgetReqSafe(J,"drivetrain.min_torque"));

  // --- Steering system ---
  P.add("natural_frequency_steering_system", JgetReqSafe(J,"steering_system.natural_frequency_steering_system"));
  P.add("steering_system_damping",           JgetReqSafe(J,"steering_system.steering_system_damping"));
  P.add("max_steer",                         JgetReqSafe(J,"steering_system.max_steer"));
  P.add("min_steer",                         JgetReqSafe(J,"steering_system.min_steer"));
  P.add("max_steering_angle_rate",         JgetReqSafe(J,"steering_system.max_steering_angle_rate"));
  P.add("min_steering_angle_rate",         JgetReqSafe(J,"steering_system.min_steering_angle_rate"));

  // --- Simulation ---
  P.add("simulation_time_step", JgetReqSafe(J,"simulation.time_step"));

  // --- DV board and PID (zgodnie z configiem params_default_lem.json) ---
  P.add("dv_board_to_maxon_time_step",        JgetReqSafe(J,"dv_board.dv_board_to_maxon_time_step"));
  P.add("control_to_dv_boad_read_time_step",  JgetReqSafe(J,"dv_board.control_to_dv_boad_read_time_step"));
  P.add("wheel_encoder_reading_time_step",    JgetReqSafe(J,"dv_board.wheel_encoder_reading_time_step"));
  P.add("dv_board_tractive_system_time_step", JgetReqSafe(J,"dv_board.dv_board_tractive_system_time_step"));

  // W configu masz pid_speed_* oraz pid_traction_*
  P.add("pid_speed_p",   JgetReqSafe(J,"dv_board.pid_speed_p"));
  P.add("pid_speed_i",   JgetReqSafe(J,"dv_board.pid_speed_i"));
  P.add("pid_speed_d",   JgetReqSafe(J,"dv_board.pid_speed_d"));
  P.add("pid_speed_max", JgetReqSafe(J,"dv_board.pid_speed_max"));
  P.add("pid_speed_min", JgetReqSafe(J,"dv_board.pid_speed_min"));


  P.add("pid_fx_target_p", JgetOpt(J, "dv_board.pid_fx_target_p", 0.0));
  P.add("pid_fx_target_i", JgetOpt(J, "dv_board.pid_fx_target_i", 0.0));
  P.add("pid_fx_target_d", JgetOpt(J, "dv_board.pid_fx_target_d", 0.0));
  P.add("pid_fx_target_max", JgetOpt(J, "dv_board.pid_fx_target_max", 1000.0));
  P.add("pid_fx_target_min", JgetOpt(J, "dv_board.pid_fx_target_min", -1000.0));
  P.add("pid_fx_target_anti_windup_gain", JgetOpt(J, "dv_board.pid_fx_target_anti_windup_gain", 0.0));
  P.add("pid_fx_target_leak_time_scale", JgetOpt(J, "dv_board.pid_fx_target_leak_time_scale", 0.0));


  P.add("pid_mz_target_p", JgetOpt(J, "dv_board.pid_mz_target_p", 0.0));
  P.add("pid_mz_target_i", JgetOpt(J, "dv_board.pid_mz_target_i", 0.0));
  P.add("pid_mz_target_d", JgetOpt(J, "dv_board.pid_mz_target_d", 0.0));
  P.add("pid_mz_target_max", JgetOpt(J, "dv_board.pid_mz_target_max", 500.0));
  P.add("pid_mz_target_min", JgetOpt(J, "dv_board.pid_mz_target_min", -500.0));
  P.add("pid_mz_target_anti_windup_gain", JgetOpt(J, "dv_board.pid_mz_target_anti_windup_gain", 0.0));
  P.add("pid_mz_target_leak_time_scale", JgetOpt(J, "dv_board.pid_mz_target_leak_time_scale", 0.0));

  // Torque allocation
  P.add("torque_allocation_lambda_factor", JgetOpt(J, "dv_board.torque_allocation_lambda_factor", 0.075));
  P.add("torque_allocation_fx_ref", JgetOpt(J, "dv_board.torque_allocation_fx_ref", 1000.0));
  P.add("torque_allocation_mz_ref", JgetOpt(J, "dv_board.torque_allocation_mz_ref", 10.0));

  // Nazwa w ParamBank musi odpowiadać temu, co używa sim_loop.cpp ("pid_scale")
  // a wartość pochodzi z JSON: dv_board.pid_speed_scale
  P.add("pid_speed_scale",           JgetReqSafe(J,"dv_board.pid_speed_scale"));

  // Traction control PID
  P.add("pid_traction_p", JgetReqSafe(J,"dv_board.pid_traction_p"));
  P.add("pid_traction_i", JgetReqSafe(J,"dv_board.pid_traction_i"));
  P.add("pid_traction_d", JgetReqSafe(J,"dv_board.pid_traction_d"));

  // Limity TC w configu są rozdzielone drive/brake
  P.add("pid_traction_max_drive", JgetReqSafe(J,"dv_board.pid_traction_max_drive"));
  P.add("pid_traction_min_drive", JgetReqSafe(J,"dv_board.pid_traction_min_drive"));
  P.add("pid_traction_max_brake", JgetReqSafe(J,"dv_board.pid_traction_max_brake"));
  P.add("pid_traction_min_brake", JgetReqSafe(J,"dv_board.pid_traction_min_brake"));

  // Dodatkowe parametry dla PID trakcji
  P.add("pid_traction_anti_windup_gain_drive", JgetReqSafe(J,"dv_board.pid_traction_anti_windup_gain_drive"));
  P.add("pid_traction_anti_windup_gain_brake", JgetReqSafe(J,"dv_board.pid_traction_anti_windup_gain_brake"));
  P.add("pid_traction_leak_time_scale_drive",  JgetReqSafe(J,"dv_board.pid_traction_leak_time_scale_drive"));
  P.add("pid_traction_leak_time_scale_brake",  JgetReqSafe(J,"dv_board.pid_traction_leak_time_scale_brake"));

  // Slip targets + progi histerezy (on/off)
  P.add("target_slip_drive", JgetReqSafe(J,"dv_board.target_slip_drive"));
  P.add("target_slip_brake", JgetReqSafe(J,"dv_board.target_slip_brake"));
  P.add("slip_drive_on",     JgetReqSafe(J,"dv_board.slip_drive_on"));
  P.add("slip_drive_off",    JgetReqSafe(J,"dv_board.slip_drive_off"));
  P.add("slip_brake_on",     JgetReqSafe(J,"dv_board.slip_brake_on"));
  P.add("slip_brake_off",    JgetReqSafe(J,"dv_board.slip_brake_off"));



    //// --- Pose / INS ---
    P.add("pose_noise",        JgetReqSafe(J,"pose.pose_noise"));
    P.add("orientation_noise", JgetReqSafe(J,"pose.orientation_noise"));
    P.add("rotation_noise",    JgetReqSafe(J,"pose.rotation_noise"));
    P.add("ins_frequancy",     JgetReqSafe(J,"pose.ins_frequancy"));
    P.add("gps_speed_frequancy", JgetReqSafe(J,"pose.gps_speed_frequancy"));
    P.add("gps_speed_noise",   JgetReqSafe(J,"pose.speed_gps_noise"));
  
    // --- Dodatkowe sensory pose (gyro/acc/gps) ---
    P.add("gyro_noise_std",         JgetReqSafe(J,"pose.gyro_noise"));
    P.add("acc_noise_std",          JgetReqSafe(J,"pose.acc_noise"));
    P.add("acc_frequancy",      JgetReqSafe(J,"pose.acc_frequancy"));
    P.add("gyro_frequancy",     JgetReqSafe(J,"pose.gyro_frequancy"));
    P.add("gps_frequancy",      JgetReqSafe(J,"pose.gps_frequancy"));
    P.add("gps_position_noise",          JgetReqSafe(J,"pose.gps_noise"));
    P.add("gps_yaw_noise",   JgetReqSafe(J,"pose.gps_heding_noise"));
  
    P.add("calibration_time",         JgetReqSafe(J,"pose.calibration_time"));
    P.add("acc_bias_drift_time_sec",  JgetReqSafe(J,"pose.acc_bias_drift_time_sec"));
    P.add("gyro_bias_drift_time_sec", JgetReqSafe(J,"pose.gyro_bias_drift_time_sec"));
    P.add("acc_bias_init_std",        JgetReqSafe(J,"pose.acc_bias_init_std"));
    P.add("gyro_bias_init_std",       JgetReqSafe(J,"pose.gyro_bias_init_std"));
    P.add("gyro_bias_rw" , P.get("gyro_bias_init_std")/std::sqrt(P.get("gyro_bias_drift_time_sec"))); // z dupy
    P.add("acc_bias_rw" ,  P.get("acc_bias_init_std")/std::sqrt(P.get("acc_bias_init_std"))); // z dupy 
    
  // --- Vision ---
  P.add("fov_x",                         JgetReqSafe(J,"vision.fov_x"));
  P.add("fov_y",                         JgetReqSafe(J,"vision.fov_y"));
  P.add("z_camera_to_cog",               JgetReqSafe(J,"vision.z_camera_to_cog"));
  P.add("x_camera_to_cog",               JgetReqSafe(J,"vision.x_camera_to_cog"));
  P.add("y_camera_to_cog",               JgetReqSafe(J,"vision.y_camera_to_cog"));
  P.add("cone_height",                   JgetReqSafe(J,"vision.cone_height"));
  P.add("frames_per_second",             JgetReqSafe(J,"vision.frames_per_second"));
  P.add("mean_time_of_vision_execuction",JgetReqSafe(J,"vision.mean_time_of_vision_execuction"));
  P.add("var_of_vision_time_execution",  JgetReqSafe(J,"vision.var_of_vision_time_execution"));
  P.add("camera_range",                  JgetReqSafe(J,"vision.camera_range"));
  P.add("vision_noise_a",                JgetReqSafe(J,"vision.vision_noise_a"));
  P.add("vision_noise_b",                JgetReqSafe(J,"vision.vision_noise_b"));

  // --- Metrics ---
  P.add("kappa_slip_threshold", JgetReqSafe(J,"metrics.kappa_slip_threshold"));
  P.add("slip_angle_slip_threshold", JgetReqSafe(J,"metrics.slip_angle_slip_threshold"));
  P.add("slip_body_slip_threshold", JgetReqSafe(J,"metrics.slip_body_slip_threshold"));
  P.add("max_yaw_rate_factor_violation", JgetReqSafe(J,"metrics.max_yaw_rate_factor_violation"));
  P.add("ey_avg_crash_threshold", JgetReqSafe(J,"metrics.ey_avg_crash_threshold"));
  P.add("vs_avg_crash_threshold", JgetReqSafe(J,"metrics.vs_avg_crash_threshold"));
  P.add("track_width", JgetReqSafe(J,"metrics.track_width"));
  P.add("max_track_violation_critical_crash", JgetReqSafe(J,"metrics.max_track_violation_critical_crash"));
  P.add("max_track_violation_high", JgetReqSafe(J,"metrics.max_track_violation_high"));
  P.add("max_track_violation_medium", JgetReqSafe(J,"metrics.max_track_violation_medium"));
  P.add("max_track_violation_soft", JgetReqSafe(J,"metrics.max_track_violation_soft"));


  return P;
}

} // namespace lem_dynamics_sim_
