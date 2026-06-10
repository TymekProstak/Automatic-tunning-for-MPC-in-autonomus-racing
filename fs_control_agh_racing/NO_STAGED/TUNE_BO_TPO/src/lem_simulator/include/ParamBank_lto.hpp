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


namespace lto {

//// ============================================================
////  Struktura głównego banku parametrów
//// ============================================================
struct ParamBank_lto {
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
inline ParamBank_lto build_param_bank(const nlohmann::json& J) {
  ParamBank_lto P;


  //// --- Lap Time Optimizer (LTO) Parameters ---
  P.add("lto_g", JgetReq(J, "lap_time_optimizer.lto_params.g"));
  P.add("lto_v_min", JgetReq(J, "lap_time_optimizer.lto_params.v_min"));
  P.add("lto_v_max", JgetReq(J, "lap_time_optimizer.lto_params.v_max"));
  P.add("lto_lf", JgetReq(J, "lap_time_optimizer.lto_params.lf"));
  P.add("lto_lr", JgetReq(J, "lap_time_optimizer.lto_params.lr"));
  P.add("lto_ds" , JgetReq(J,"lap_time_optimizer.lto_params.ds"));

  P.add("lto_Cm", JgetReq(J, "lap_time_optimizer.lto_params.Cm"));
  P.add("lto_Cr0", JgetReq(J, "lap_time_optimizer.lto_params.Cr0"));
  P.add("lto_Cl", JgetReq(J, "lap_time_optimizer.lto_params.Cl"));
  P.add("lto_Cd", JgetReq(J, "lap_time_optimizer.lto_params.Cd"));

  P.add("lto_max_drive_power", JgetReq(J, "lap_time_optimizer.lto_params.max_drive_power"));
  P.add("lto_max_brake_power", JgetReq(J, "lap_time_optimizer.lto_params.max_brake_power"));


  P.add("lto_max_delta", JgetReq(J, "lap_time_optimizer.lto_params.max_delta"));
  P.add("lto_min_delta", JgetReq(J, "lap_time_optimizer.lto_params.min_delta"));

  P.add("lto_max_d_delta", JgetReq(J, "lap_time_optimizer.lto_params.max_d_delta"));
  P.add("lto_min_d_delta", JgetReq(J, "lap_time_optimizer.lto_params.min_d_delta"));

  P.add("lto_max_d_T", JgetReq(J, "lap_time_optimizer.lto_params.max_d_T"));
  P.add("lto_min_d_T", JgetReq(J, "lap_time_optimizer.lto_params.min_d_T"));
  P.add("lto_max_tv" , JgetReq(J, "lap_time_optimizer.lto_params.lto_max_tv"));
  P.add("lto_min_tv" , JgetReq(J, "lap_time_optimizer.lto_params.lto_min_tv"));

  P.add("lto_Fz_nom", JgetReq(J, "lap_time_optimizer.lto_params.Fz_nom"));

  P.add("lto_mu_y", JgetReq(J, "lap_time_optimizer.lto_params.mu_y"));
  P.add("lto_mu_x", JgetReq(J, "lap_time_optimizer.lto_params.mu_x"));

  P.add("lto_C", JgetReq(J, "lap_time_optimizer.lto_params.C"));
  P.add("lto_D", JgetReq(J, "lap_time_optimizer.lto_params.D"));
  P.add("lto_B", JgetReq(J, "lap_time_optimizer.lto_params.B"));
  
  P.add("lto_length", JgetReq(J, "lap_time_optimizer.lto_params.length"));
  P.add("lto_width", JgetReq(J, "lap_time_optimizer.lto_params.width"));
  P.add("lto_track_width", JgetReq(J, "lap_time_optimizer.lto_params.track_width"));

  P.add("lto_d_delta_cost", JgetReq(J, "lap_time_optimizer.lto_params.d_delta_cost"));
  P.add("lto_d_T_cost", JgetReq(J, "lap_time_optimizer.lto_params.d_T_cost"));
  P.add("lto_beta_cost", JgetReq(J, "lap_time_optimizer.lto_params.beta_cost"));
  P.add("lto_s_dot_cost", JgetReq(J, "lap_time_optimizer.lto_params.s_dot_cost"));
  P.add("lto_tv_cost" , JgetReq(J, "lap_time_optimizer.lto_params.lto_tv_cost"));

  P.add("lto_m", JgetReq(J, "lap_time_optimizer.lto_params.m"));
  P.add("lto_Iz", JgetReq(J, "lap_time_optimizer.lto_params.Iz"));
  P.add("lto_saftey_factor", JgetReq(J, "lap_time_optimizer.lto_params.saftey_factor")); 

  return P;
}

} // namespace v2_control
