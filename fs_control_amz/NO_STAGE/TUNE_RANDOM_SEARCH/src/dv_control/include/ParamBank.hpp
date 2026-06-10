#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace v2_control {

//// ============================================================
////  ParamBank — trzymam wszystkie parametry jako double
//// ============================================================
struct ParamBank {
  std::vector<std::string> names;
  std::unordered_map<std::string, int> idx;
  std::vector<double> values;

  int add(const std::string &name, double val) {
    auto it = idx.find(name);
    if (it != idx.end()) {
      values[it->second] = val;
      return it->second;
    }
    const int k = static_cast<int>(names.size());
    names.push_back(name);
    idx[name] = k;
    values.push_back(val);
    return k;
  }

  bool has(const std::string &name) const { return idx.find(name) != idx.end(); }

  int i(const std::string &name) const {
    auto it = idx.find(name);
    if (it == idx.end()) {
      throw std::runtime_error("ParamBank: missing key '" + name + "'");
    }
    return it->second;
  }

  double get(const std::string &name) const { return values.at(i(name)); }

  double getOr(const std::string &name, double def) const {
    return has(name) ? get(name) : def;
  }

  void set(const std::string &name, double v) { values.at(i(name)) = v; }

  size_t size() const { return values.size(); }
};

//// ============================================================
////  JSON helpers — czytam ścieżki typu "mpc.model.m"
//// ============================================================

inline const nlohmann::json &JnodeReq(const nlohmann::json &J, const std::string &path) {
  const auto pos = path.find('.');
  if (pos == std::string::npos) {
    if (!J.contains(path)) {
      throw std::runtime_error("JSON: missing required key '" + path + "'");
    }
    return J.at(path);
  }

  const std::string head = path.substr(0, pos);
  const std::string tail = path.substr(pos + 1);

  if (!J.contains(head)) {
    throw std::runtime_error("JSON: missing object '" + head + "'");
  }
  return JnodeReq(J.at(head), tail);
}

inline const nlohmann::json *JnodeOpt(const nlohmann::json &J, const std::string &path) {
  const auto pos = path.find('.');
  if (pos == std::string::npos) {
    if (!J.contains(path)) return nullptr;
    return &J.at(path);
  }

  const std::string head = path.substr(0, pos);
  const std::string tail = path.substr(pos + 1);

  if (!J.contains(head)) return nullptr;
  return JnodeOpt(J.at(head), tail);
}

template <typename T>
inline T JgetReqT(const nlohmann::json &J, const std::string &path) {
  return JnodeReq(J, path).get<T>();
}

template <typename T>
inline T JgetReqSafeT(const nlohmann::json &J, const std::string &path) {
  try {
    return JgetReqT<T>(J, path);
  } catch (const std::exception &e) {
    std::cerr << "\n[JSON ERROR] while reading required path: " << path
              << "\n  what(): " << e.what() << "\n"
              << std::endl;
    throw;
  }
}

template <typename T>
inline T JgetOptT(const nlohmann::json &J, const std::string &path, const T &def) {
  const nlohmann::json *node = JnodeOpt(J, path);
  if (!node) return def;
  return node->get<T>();
}

template <typename T>
inline T JgetOptSafeT(const nlohmann::json &J, const std::string &path, const T &def) {
  try {
    return JgetOptT<T>(J, path, def);
  } catch (const std::exception &e) {
    std::cerr << "\n[JSON ERROR] while reading optional path: " << path
              << "\n  what(): " << e.what()
              << "\n  using default: " << def << "\n"
              << std::endl;
    throw;
  }
}

inline nlohmann::json load_json_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + path);
  }
  nlohmann::json J;
  f >> J;
  return J;
}

//// ============================================================
////  Buduję bank parametrów z JSON
//// ============================================================
inline ParamBank build_param_bank(const nlohmann::json &J) {
  ParamBank P;

  auto add_req = [&](const char *bank_key, const char *json_path) {
    P.add(bank_key, JgetReqSafeT<double>(J, json_path));
  };

  auto add_req_alias = [&](const char *json_path, std::initializer_list<const char *> bank_keys) {
    const double v = JgetReqSafeT<double>(J, json_path);
    for (const auto *k : bank_keys) {
      P.add(k, v);
    }
  };

  auto add_req_bool = [&](const char *bank_key, const char *json_path) {
    const bool b = JgetReqSafeT<bool>(J, json_path);
    P.add(bank_key, b ? 1.0 : 0.0);
  };

  auto add_req_bool_alias = [&](const char *json_path,
                                std::initializer_list<const char *> bank_keys) {
    const bool b = JgetReqSafeT<bool>(J, json_path);
    const double v = b ? 1.0 : 0.0;
    for (const auto *k : bank_keys) {
      P.add(k, v);
    }
  };

  // ==========================================================
  // Stanley
  // ==========================================================
  add_req_alias("stanley.k",       {"stanley_k"});
  add_req_alias("stanley.epsilon", {"stanley_epsilon"});
  add_req_alias("stanley.lf",      {"stanley_lf"});

  // ==========================================================
  // General
  // ==========================================================
  add_req_alias("general.max_delta", {"general_max_delta", "max_delta"});
  add_req_alias("general.min_delta", {"general_min_delta", "min_delta"});

  add_req_alias("general.odom_frequency", {"general_odom_frequency", "odom_frequency"});

  add_req("general_interpoleted_num_max_points",
          "general.interpoleted_num_max_points");
  add_req("general_distance_between_interpoleted_points",
          "general.distance_between_interpoleted_points");

  add_req_alias("general.v_target", {"general_v_target", "v_target"});
  add_req("general_min_path_length_for_geo", "general.min_path_length_for_geo");

  add_req_bool_alias("general.using_stanley",
                     {"using_stanley", "general_using_stanley"});

  // ==========================================================
  // Velocity planner
  // ==========================================================
  add_req_alias("velocity_planner.v_min", {"vel_planner_v_min"});
  add_req_alias("velocity_planner.v_max", {"vel_planner_v_max"});

  add_req_alias("velocity_planner.mux_acc", {"vel_planner_mux_acc"});
  add_req_alias("velocity_planner.mux_dec", {"vel_planner_mux_dec"});
  add_req_alias("velocity_planner.muy",     {"vel_planner_muy"});
  add_req_alias("velocity_planner.safety_factor", {"vel_planner_safety_factor"});

  add_req_alias("velocity_planner.max_jerk", {"vel_planner_max_jerk"});
  add_req_alias("velocity_planner.spatial_step", {"vel_planner_spatial_step"});
  add_req_alias("velocity_planner.number_of_jerk_merging_iterations",
                {"vel_planner_number_of_jerk_merging_iterations"});
  add_req_alias("velocity_planner.smoothing_factor", {"vel_planner_smoothing_factor"});
  add_req_alias("velocity_planner.Cl", {"vel_planner_Cl"});
  add_req_alias("velocity_planner.m",  {"vel_planner_m"});

  // ==========================================================
  // MPC solver meta
  // ==========================================================
  add_req_alias("mpc.solver.mpc_N",  {"mpc_solver_mpc_N", "mpc_N"});
  add_req_alias("mpc.solver.mpc_dt", {"mpc_solver_mpc_dt", "mpc_dt"});

  add_req("mpc_solver_n_sqp",      "mpc.solver.n_sqp");
  add_req("mpc_solver_sqp_mixing", "mpc.solver.sqp_mixing");
  add_req("mpc_solver_n_reset",    "mpc.solver.n_reset");
  add_req("mpc_solver_eps",        "mpc.solver.eps");

  add_req("mpc_solver_sim_stages", "mpc.solver.sim_stages");
  add_req("mpc_solver_sim_steps",  "mpc.solver.sim_steps");

  add_req("mpc_solver_s_trust_region", "mpc.solver.s_trust_region");
  add_req("mpc_solver_reg_epsilon",    "mpc.solver.reg_epsilon");

  add_req_bool("mpc_solver_hard_reset_before_every_solve",
               "mpc.solver.hard_reset_before_every_solve");
  add_req_bool("mpc_solver_hard_reset_after_fail",
               "mpc.solver.hard_reset_after_fail");

  // ==========================================================
  // MPC bounds
  // ==========================================================
  add_req_alias("mpc.bounds.min_ey",    {"mpc_bounds_min_ey"});
  add_req_alias("mpc.bounds.max_ey",    {"mpc_bounds_max_ey"});

  add_req_alias("mpc.bounds.min_epsi",  {"mpc_bounds_min_epsi"});
  add_req_alias("mpc.bounds.max_epsi",  {"mpc_bounds_max_epsi"});

  add_req_alias("mpc.bounds.min_vx",    {"mpc_bounds_min_vx"});
  add_req_alias("mpc.bounds.max_vx",    {"mpc_bounds_max_vx"});

  add_req_alias("mpc.bounds.min_vy",    {"mpc_bounds_min_vy"});
  add_req_alias("mpc.bounds.max_vy",    {"mpc_bounds_max_vy"});

  add_req_alias("mpc.bounds.min_r",     {"mpc_bounds_min_r"});
  add_req_alias("mpc.bounds.max_r",     {"mpc_bounds_max_r"});

  add_req_alias("mpc.bounds.min_delta", {"mpc_bounds_min_delta"});
  add_req_alias("mpc.bounds.max_delta", {"mpc_bounds_max_delta"});

  add_req_alias("mpc.bounds.min_T",     {"mpc_bounds_min_T"});
  add_req_alias("mpc.bounds.max_T",     {"mpc_bounds_max_T"});

  add_req_alias("mpc.bounds.min_ddelta_state", {"mpc_bounds_min_ddelta_state"});
  add_req_alias("mpc.bounds.max_ddelta_state", {"mpc_bounds_max_ddelta_state"});

  add_req_alias("mpc.bounds.min_delta_cmd_state", {"mpc_bounds_min_delta_cmd_state"});
  add_req_alias("mpc.bounds.max_delta_cmd_state", {"mpc_bounds_max_delta_cmd_state"});

  add_req_alias("mpc.bounds.min_u_ddelta_cmd", {"mpc_bounds_min_u_ddelta_cmd"});
  add_req_alias("mpc.bounds.max_u_ddelta_cmd", {"mpc_bounds_max_u_ddelta_cmd"});

  add_req_alias("mpc.bounds.min_dT",  {"mpc_bounds_min_dT"});
  add_req_alias("mpc.bounds.max_dT",  {"mpc_bounds_max_dT"});

  add_req_alias("mpc.bounds.min_Mtv", {"mpc_bounds_min_Mtv"});
  add_req_alias("mpc.bounds.max_Mtv", {"mpc_bounds_max_Mtv"});

  // ==========================================================
  // MPC constraints
  // ==========================================================
  add_req_alias("mpc.constraints.safety_factor",
                {"mpc_constraints_safety_factor", "safety_factor"});

  add_req_alias("mpc.constraints.track_width",
                {"mpc_constraints_track_width", "track_width"});

  add_req_alias("mpc.constraints.L_c",
                {"mpc_constraints_L_c", "L_c"});

  add_req_alias("mpc.constraints.W_c",
                {"mpc_constraints_W_c", "W_c"});

  // ==========================================================
  // MPC model
  // ==========================================================
  add_req_alias("mpc.model.m",  {"model_m", "mpc_model_m"});
  add_req_alias("mpc.model.Iz", {"model_Iz", "mpc_model_Iz"});
  add_req_alias("mpc.model.lf", {"model_lf", "mpc_model_lf"});
  add_req_alias("mpc.model.lr", {"model_lr", "mpc_model_lr"});

  add_req_alias("mpc.model.Cd",  {"model_Cd", "mpc_model_Cd"});
  add_req_alias("mpc.model.Cr0", {"model_Cr0", "mpc_model_Cr0"});
  add_req_alias("mpc.model.Cl",  {"model_Cl", "mpc_model_Cl"});
  add_req_alias("mpc.model.Cm",  {"model_Cm", "mpc_model_Cm"});

  add_req_alias("mpc.model.B", {"model_B", "mpc_model_B"});
  add_req_alias("mpc.model.C", {"model_C", "mpc_model_C"});
  add_req_alias("mpc.model.D", {"model_D", "mpc_model_D"});

  add_req_alias("mpc.model.mux", {"model_mux", "mpc_model_mux"});
  add_req_alias("mpc.model.muy", {"model_muy", "mpc_model_muy"});

  add_req_alias("mpc.model.vx_eps", {"model_vx_eps", "mpc_model_vx_eps"});

  add_req_alias("mpc.model.car_length", {"model_car_length", "mpc_model_car_length"});
  add_req_alias("mpc.model.car_width",  {"model_car_width",  "mpc_model_car_width"});

  add_req_alias("mpc.model.steer_natural_freq",
                {"model_steer_natural_freq", "mpc_model_steer_natural_freq"});
  add_req_alias("mpc.model.steer_damping",
                {"model_steer_damping", "mpc_model_steer_damping"});

  add_req_alias("mpc.model.max_steering_angle_rate",
                {"model_max_steering_angle_rate", "mpc_model_max_steering_angle_rate"});
  add_req_alias("mpc.model.min_steering_angle_rate",
                {"model_min_steering_angle_rate", "mpc_model_min_steering_angle_rate"});

  add_req_alias("mpc.model.wheel_radius",
                {"model_wheel_radius", "mpc_model_wheel_radius"});

  add_req_alias("mpc.model.max_motor_torque",
                {"model_max_motor_torque", "mpc_model_max_motor_torque"});

  // ==========================================================
  // Effective mu (DERIVED)
  // ==========================================================
  {
    const double sf       = P.get("mpc_constraints_safety_factor");
    const double mux_base = P.get("model_mux");
    const double muy_base = P.get("model_muy");

    P.add("mux_effective", std::max(1e-6, mux_base * sf));
    P.add("muy_effective", std::max(1e-6, muy_base * sf));
  }

  // ==========================================================
  // MPC / Frenet cost
  // ==========================================================
  add_req_alias("mpc.cost.q_ey",   {"mpc_cost_q_ey", "q_ey"});
  add_req_alias("mpc.cost.q_sdot", {"mpc_cost_q_sdot", "q_sdot"});

  add_req_alias("mpc.cost.R_delta", {"mpc_cost_R_delta", "R_delta"});
  add_req_alias("mpc.cost.R_T",     {"mpc_cost_R_T", "R_T"});

  add_req_alias("mpc.cost.R_u_ddelta_cmd",
                {"mpc_cost_R_u_ddelta_cmd", "R_u_ddelta_cmd"});
  add_req_alias("mpc.cost.R_dT",
                {"mpc_cost_R_dT", "R_dT"});
  add_req_alias("mpc.cost.R_Mtv",
                {"mpc_cost_R_Mtv", "R_Mtv"});

  add_req_alias("mpc.cost.Q_vy",
                {"mpc_cost_Q_vy", "Q_vy"});
  add_req_alias("mpc.cost.Q_ddelta_state",
                {"mpc_cost_Q_ddelta_state", "Q_ddelta_state"});
  add_req_alias("mpc.cost.Q_delta_cmd_state",
                {"mpc_cost_Q_delta_cmd_state", "Q_delta_cmd_state"});

  add_req_alias("mpc.cost.Q_beta",
                {"mpc_cost_Q_beta", "Q_beta"});
  add_req_alias("mpc.cost.R_ddelta",
                {"mpc_cost_R_ddelta", "R_ddelta"});

 // add_req_alias("mpc.cost.Q_beta_dyn_kin",
     //           {"mpc_cost_Q_beta_dyn_kin", "Q_beta_dyn_kin"});
  add_req_alias("mpc.cost.Q_epsi",
                {"mpc_cost_Q_epsi", "Q_epsi"});
  add_req_alias("mpc.cost.Q_yaw_rate",
                {"mpc_cost_Q_yaw_rate", "Q_yaw_rate"});

  add_req_alias("mpc.cost.q_ey_terminal",
                {"mpc_cost_q_ey_terminal", "q_ey_terminal"});
  add_req_alias("mpc.cost.Q_epsi_terminal",
                {"mpc_cost_Q_epsi_terminal", "Q_epsi_terminal"});
  add_req_alias("mpc.cost.Q_vx_terminal",
                {"mpc_cost_Q_vx_terminal", "Q_vx_terminal"});
  add_req_alias("mpc.cost.Q_vy_terminal",
                {"mpc_cost_Q_vy_terminal", "Q_vy_terminal"});
  add_req_alias("mpc.cost.Q_yaw_rate_terminal",
                {"mpc_cost_Q_yaw_rate_terminal", "Q_yaw_rate_terminal"});
  add_req_alias("mpc.cost.Q_delta_terminal",
                {"mpc_cost_Q_delta_terminal", "Q_delta_terminal"});
  add_req_alias("mpc.cost.Q_ddelta_terminal",
                {"mpc_cost_Q_ddelta_terminal", "Q_ddelta_terminal"});
  add_req_alias("mpc.cost.Q_delta_cmd_terminal",
                {"mpc_cost_Q_delta_cmd_terminal", "Q_delta_cmd_terminal"});
  add_req_alias("mpc.cost.Q_T_terminal",
                {"mpc_cost_Q_T_terminal", "Q_T_terminal"});

  // ==========================================================
  // Slack weights
  // ==========================================================
  add_req_alias("mpc.cost.q_slack_track_lin",
                {"mpc_cost_q_slack_track_lin"});
  add_req_alias("mpc.cost.q_slack_track_quad",
                {"mpc_cost_q_slack_track_quad"});
  add_req_alias("mpc.cost.q_slack_fric_lin",
                {"mpc_cost_q_slack_fric_lin"});
  add_req_alias("mpc.cost.q_slack_fric_quad",
                {"mpc_cost_q_slack_fric_quad"});

  // ==========================================================
  // Runtime helpers
  // ==========================================================
  {
    const double vmax = P.get("vel_planner_v_max");
    P.add("mpc_runtime_vmax", vmax);
  }

  return P;
}

inline ParamBank build_param_bank_from_file(const std::string &json_path) {
  return build_param_bank(load_json_file(json_path));
}

} // namespace v2_control
