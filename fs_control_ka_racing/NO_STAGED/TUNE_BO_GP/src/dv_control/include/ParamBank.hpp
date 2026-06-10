#pragma once
#include <Eigen/Dense>


#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>


namespace v2_control {

//// ============================================================
////  Struktura głównego banku parametrów
//// ============================================================
struct ParamBank {
  std::vector<std::string> names;                 // nazwy w kolejności
  std::unordered_map<std::string, int> idx;       // nazwa -> index
  std::vector<double> values;                     // wartości liczbowe

  int add(const std::string& name, double val) {
    auto it = idx.find(name);
    if (it != idx.end()) {
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
    if (it == idx.end()) throw std::runtime_error("ParamBank: missing key '" + name + "'");
    return it->second;
  }

  double get(const std::string& name) const { return values.at(i(name)); }
  void set(const std::string& name, double v) { values.at(i(name)) = v; }
  size_t size() const { return values.size(); }
};

//// ============================================================
////  Pomocnicze funkcje do pobierania z JSON-a
//// ============================================================

// --- Wymagane pole ---
inline double JgetReq(const nlohmann::json& J, const std::string& path) {
  auto pos = path.find('.');
  if (pos == std::string::npos) {
    if (!J.contains(path))
      throw std::runtime_error("JSON: missing required key '" + path + "'");
    return J.at(path).get<double>();
  }
  std::string head = path.substr(0, pos);
  std::string tail = path.substr(pos + 1);
  if (!J.contains(head))
    throw std::runtime_error("JSON: missing object '" + head + "'");
  return JgetReq(J.at(head), tail);
}

// --- Bezpieczna wersja z diagnostyką ---
inline double JgetReqSafe(const nlohmann::json& J, const std::string& path) {
  try {
    return JgetReq(J, path);
  } catch (const std::exception& e) {
    std::cerr << "\n[JSON ERROR] while reading path: " << path
              << "\n  what(): " << e.what() << "\n" << std::endl;
    throw;  // przekazujemy wyjątek dalej, by zatrzymać node
  }
}

// --- Opcjonalne pole z wartością domyślną ---
inline double JgetOpt(const nlohmann::json& J, const std::string& path, double def) {
  try { return JgetReq(J, path); } catch (...) { return def; }
}

//// ============================================================
////  Budowa banku parametrów z pliku JSON
//// ============================================================
inline ParamBank build_param_bank(const nlohmann::json& J) {
  ParamBank P;

  
  //// --- Stanley Controller ---
  P.add("stanley_k", JgetReq(J, "stanley.k"));
  P.add("stanley_epsilon", JgetReq(J, "stanley.epsilon"));
  P.add("stanley_lf", JgetReq(J, "stanley.lf"));


  //// --- General Parameters ---
  P.add("max_delta", JgetReq(J, "general.max_delta"));
  P.add("min_delta", JgetReq(J, "general.min_delta"));
  P.add("odom_frequency", JgetReq(J, "general.odom_frequency"));
  P.add("interpoleted_num_max_points", JgetReq(J, "general.interpoleted_num_max_points"));
  P.add("distance_between_interpoleted_points", JgetReq(J, "general.distance_between_interpoleted_points"));
  P.add("v_target", JgetReq(J, "general.v_target"));
  P.add("using_stanley",JgetReq(J,"general.using_stanley"));
  P.add("min_path_length_for_geo", JgetReq(J,"general.min_path_length_for_geo"));
 
  //// --- MPC Parameters ---
  P.add("mpc_max_delta", JgetReq(J, "mpc.bounds.max_delta"));
  P.add("mpc_min_delta", JgetReq(J, "mpc.bounds.min_delta"));
  P.add("mpc_max_ddelta", JgetReq(J, "mpc.bounds.max_ddelta"));
  P.add("mpc_min_ddelta", JgetReq(J, "mpc.bounds.min_ddelta"));
  P.add("mpc_max_tv" , JgetReq(J,"mpc.bounds.max_tv"));
  P.add("mpc_min_tv" , JgetReq(J,"mpc.bounds.min_tv"));
  
  P.add("model_m", JgetReq(J, "mpc.model.m"));
  P.add("model_Iz", JgetReq(J, "mpc.model.Iz"));
  P.add("model_lf", JgetReq(J, "mpc.model.lf"));
  P.add("model_lr", JgetReq(J, "mpc.model.lr"));
  P.add("model_Cf", JgetReq(J, "mpc.model.Cf"));
  P.add("model_Cr", JgetReq(J, "mpc.model.Cr"));
  P.add("model_Cr0", JgetReq(J, "mpc.model.Cr0"));
  P.add("model_wheel_radius", JgetReq(J, "mpc.model.whell_radius"));
  P.add("model_max_motor_torque", JgetReq(J, "mpc.model.max_motor_torque"));
  P.add("model_Cd", JgetReq(J, "mpc.model.Cd"));
  P.add("model_C", JgetReq(J, "mpc.model.C"));
  P.add("model_D", JgetReq(J, "mpc.model.D"));
  P.add("model_B", JgetReq(J, "mpc.model.B"));
  
  // --- Brakujące parametry z wrapper.cpp ---
  P.add("model_steer_natural_freq", JgetReq(J, "mpc.model.steer_natural_freq"));
  P.add("model_steer_damping", JgetReq(J, "mpc.model.steer_damping"));
  P.add("model_max_steering_angle_rate", JgetReq(J, "mpc.model.max_steering_angle_rate"));
  P.add("model_mux", JgetReq(J, "mpc.model.mux"));
  P.add("model_muy", JgetReq(J, "mpc.model.muy"));

  P.add("track_width", JgetReq(J, "mpc.constraints.track_width"));

  P.add("mpc_cost_Q_y", JgetReq(J, "mpc.cost.Q_y"));
  P.add("mpc_cost_Q_psi", JgetReq(J, "mpc.cost.Q_psi"));
  P.add("mpc_cost_R_ddelta", JgetReq(J, "mpc.cost.R_ddelta"));
  P.add("mpc_cost_Q_r", JgetReq(J, "mpc.cost.Q_r"));
  P.add("mpc_cost_Q_delta", JgetReq(J, "mpc.cost.Q_delta"));
  P.add("mpc_cost_R_tv", JgetReq(J, "mpc.cost.R_tv"));
  P.add("mpc_cost_R_dthrottle", JgetReq(J, "mpc.cost.R_dthrottle"));
  P.add("mpc_cost_Q_s_dot", JgetReq(J, "mpc.cost.Q_s_dot"));
  P.add("mpc_cost_Q_beta", JgetReq(J, "mpc.cost.Q_beta"));

  P.add("mpc_max_iter", JgetReq(J, "mpc.solver.max_iter"));
  P.add("mpc_eps", JgetReq(J, "mpc.solver.eps"));
  P.add("mpc_N", JgetReq(J, "mpc.solver.mpc_N"));
  P.add("mpc_spatial_step", P.get("v_target") / P.get("odom_frequency"));
  P.add("mpc_min_horizon_meters_length", P.get("mpc_N") * P.get("mpc_spatial_step"));
  P.add("mpc_dt", 1.0 / P.get("odom_frequency"));

  // --- Velocity Planner Parameters ---

  P.add("vel_planner_mux_acc", JgetReq(J, "velocity_planner.mux_acc"));
  P.add("vel_planner_mux_dec", JgetReq(J, "velocity_planner.mux_dec"));
  P.add("vel_planner_safety_factor", JgetReq(J, "velocity_planner.safety_factor"));
  P.add("vel_planner_v_min", JgetReq(J, "velocity_planner.v_min"));
  P.add("vel_planner_v_max", JgetReq(J, "velocity_planner.v_max"));
  P.add("vel_planner_muy" , JgetReq(J, "velocity_planner.muy"));
  P.add("vel_planner_max_jerk", JgetReq(J, "velocity_planner.max_jerk"));
  P.add("vel_planner_spatial_step", JgetReq(J, "velocity_planner.spatial_step"));
  P.add("vel_planner_number_of_jerk_merging_iterations", JgetReq(J, "velocity_planner.number_of_jerk_merging_iterations"));
  P.add("vel_planner_smoothing_factor", JgetReq(J, "velocity_planner.smoothing_factor"));
  P.add("vel_planner_saftey_factor", JgetReq(J, "velocity_planner.safety_factor"));
  P.add("vel_planner_Cl", JgetReq(J, "velocity_planner.Cl"));
  P.add("vel_planner_m", JgetReq(J, "velocity_planner.m"));

 

  return P;
}

} // namespace v2_control
