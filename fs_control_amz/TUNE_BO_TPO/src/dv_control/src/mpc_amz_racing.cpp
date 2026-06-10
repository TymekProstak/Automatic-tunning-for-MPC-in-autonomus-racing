#include "mpc_interface_amz_racing.hpp"
#include "velocity_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace v2_control {

namespace {

// ============================================================
// Debug control
// ============================================================

static int g_solve_call_counter = 0;
constexpr int DEBUG_PREVIEW_FIRST_CALLS = 5;

inline double clamp_(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

inline double wrap_angle_(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

inline double vx_safe_(double vx) {
  return std::sqrt(vx * vx + 1e-8);
}

inline std::array<double, NX> make_zero_x_() {
  std::array<double, NX> x{};
  x.fill(0.0);
  return x;
}

inline std::array<double, NU> make_zero_u_() {
  std::array<double, NU> u{};
  u.fill(0.0);
  return u;
}

// ============================================================
// Param helpers -- STRICT
// ============================================================

static inline double get_mux_eff_(const ParamBank& param) {
  return param.get("mux_effective");
}

static inline double get_muy_eff_(const ParamBank& param) {
  return param.get("muy_effective");
}

static inline double get_min_state_ddelta_(const ParamBank& param) {
  return param.get("mpc_bounds_min_ddelta_state");
}

static inline double get_max_state_ddelta_(const ParamBank& param) {
  return param.get("mpc_bounds_max_ddelta_state");
}

static inline double get_min_delta_cmd_state_(const ParamBank& param) {
  return param.get("mpc_bounds_min_delta_cmd_state");
}

static inline double get_max_delta_cmd_state_(const ParamBank& param) {
  return param.get("mpc_bounds_max_delta_cmd_state");
}

static inline double get_min_u_ddelta_cmd_(const ParamBank& param) {
  return param.get("mpc_bounds_min_u_ddelta_cmd");
}

static inline double get_max_u_ddelta_cmd_(const ParamBank& param) {
  return param.get("mpc_bounds_max_u_ddelta_cmd");
}

static inline double get_q_ey_(const ParamBank& param) {
  return param.get("mpc_cost_q_ey");
}

static inline double get_q_epsi_(const ParamBank& param) {
  return param.get("mpc_cost_Q_epsi");
}

static inline double get_r_u_ddelta_cmd_(const ParamBank& param) {
  return param.get("mpc_cost_R_u_ddelta_cmd");
}

static inline double get_r_dT_(const ParamBank& param) {
  return param.get("mpc_cost_R_dT");
}

static inline double get_r_Mtv_(const ParamBank& param) {
  return param.get("mpc_cost_R_Mtv");
}

static inline double get_q_beta_dyn_kin_(const ParamBank& param) {
  return param.get("mpc_cost_Q_beta");
}

static inline double get_gamma_progress_(const ParamBank& param) {
  return param.get("mpc_cost_q_sdot");
}

static inline double get_q_ey_terminal_(const ParamBank& param) {
  return param.get("mpc_cost_q_ey_terminal");
}

static inline double get_q_epsi_terminal_(const ParamBank& param) {
  return param.get("mpc_cost_Q_epsi_terminal");
}

// ============================================================
// State helpers
// ============================================================

static inline std::array<double, NX> clamp_state_to_runtime_bounds_(
    const std::array<double, NX>& x_in,
    const ParamBank& param)
{
  std::array<double, NX> x = x_in;

  x[IX_EY]       = clamp_(x[IX_EY],       param.get("mpc_bounds_min_ey"),    param.get("mpc_bounds_max_ey"));
  x[IX_EPSI]     = clamp_(x[IX_EPSI],     param.get("mpc_bounds_min_epsi"),  param.get("mpc_bounds_max_epsi"));
  x[IX_VX]       = clamp_(x[IX_VX],       param.get("mpc_bounds_min_vx"),    param.get("mpc_bounds_max_vx"));
  x[IX_VY]       = clamp_(x[IX_VY],       param.get("mpc_bounds_min_vy"),    param.get("mpc_bounds_max_vy"));
  x[IX_R]        = clamp_(x[IX_R],        param.get("mpc_bounds_min_r"),     param.get("mpc_bounds_max_r"));
  x[IX_DELTA]    = clamp_(x[IX_DELTA],    param.get("mpc_bounds_min_delta"), param.get("mpc_bounds_max_delta"));
  x[IX_DDELTA]   = clamp_(x[IX_DDELTA],   get_min_state_ddelta_(param),      get_max_state_ddelta_(param));
  x[IX_DELTACMD] = clamp_(x[IX_DELTACMD], get_min_delta_cmd_state_(param),   get_max_delta_cmd_state_(param));
  x[IX_T]        = clamp_(x[IX_T],        param.get("mpc_bounds_min_T"),     param.get("mpc_bounds_max_T"));

  return x;
}

inline std::array<double, NX> measured_cartesian_to_frenet_(
    const MPCC_State& x_cart,
    const TrackSpline2D& track,
    double s0_wrapped)
{
  std::array<double, NX> x = make_zero_x_();

  const double cx  = track.getX(s0_wrapped);
  const double cy  = track.getY(s0_wrapped);
  const double yaw = track.getYaw(s0_wrapped);

  const double dx = x_cart.X - cx;
  const double dy = x_cart.Y - cy;

  const double ey   = -std::sin(yaw) * dx + std::cos(yaw) * dy;
  const double epsi = wrap_angle_(x_cart.phi - yaw);

  x[IX_EY]       = ey;
  x[IX_EPSI]     = epsi;
  x[IX_VX]       = x_cart.vx;
  x[IX_VY]       = x_cart.vy;
  x[IX_R]        = x_cart.r;
  x[IX_DELTA]    = x_cart.delta;
  x[IX_DDELTA]   = x_cart.delta_dot;
  x[IX_DELTACMD] = x_cart.delta_request;
  x[IX_T]        = x_cart.T;

  return x;
}

inline void frenet_to_cartesian_(
    const TrackSpline2D& track,
    double s_wrapped,
    double ey,
    double& X,
    double& Y)
{
  const double cx  = track.getX(s_wrapped);
  const double cy  = track.getY(s_wrapped);
  const double yaw = track.getYaw(s_wrapped);

  X = cx - std::sin(yaw) * ey;
  Y = cy + std::cos(yaw) * ey;
}

inline double planner_vref_at_s_(
    const VelocityPlannerResult& vp,
    double s_wrapped,
    double fallback_vx,
    const ParamBank& param)
{
  double vref = fallback_vx;

  if (vp.valid) {
    vref = vp.v_on_spline_interpolate(s_wrapped);
  }

  return clamp_(vref,
                param.get("mpc_bounds_min_vx"),
                param.get("mpc_bounds_max_vx"));
}

// ============================================================
// IMPORTANT:
// This is kept consistent with Python codegen:
// s_dot = (vx*cos(epsi) - vy*sin(epsi)) / (1 - ey*kappa)
// ============================================================

inline double sdot_from_state_(
    const std::array<double, NX>& x,
    double kappa)
{
  return (x[IX_VX] * std::cos(x[IX_EPSI]) - x[IX_VY] * std::sin(x[IX_EPSI]))
         / (1.0 - x[IX_EY] * kappa);
}

inline void f_ode_frenet_(
    const std::array<double, NX>& x,
    const std::array<double, NU>& u,
    double kappa,
    const ParamBank& param,
    std::array<double, NX>& dx)
{
  const double m   = param.get("model_m");
  const double lf  = param.get("model_lf");
  const double lr  = param.get("model_lr");
  const double Iz  = param.get("model_Iz");

  const double B   = param.get("model_B");
  const double C_  = param.get("model_C");
  const double D_  = param.get("model_D");

  const double Cr0 = param.get("model_Cr0");
  const double Cd  = param.get("model_Cd");
  const double Cl  = param.get("model_Cl");
  const double Cm  = param.get("model_Cm");

  const double omega_n = param.get("model_steer_natural_freq");
  const double zeta    = param.get("model_steer_damping");

  const double ey          = x[IX_EY];
  const double epsi        = x[IX_EPSI];
  const double vx          = x[IX_VX];
  const double vy          = x[IX_VY];
  const double r           = x[IX_R];
  const double delta       = x[IX_DELTA];
  const double ddelta      = x[IX_DDELTA];
  const double delta_cmd   = x[IX_DELTACMD];
  const double T           = x[IX_T];

  const double u_ddelta_cmd = u[IU_DDELTACMD];
  const double dT           = u[IU_DT];
  const double Mtv          = u[IU_MTV];

  const double vx_s = vx_safe_(vx);
  const double sdot_roll = (vx * std::cos(epsi) - vy * std::sin(epsi)) / (1.0 - ey * kappa);

  const double FN_net = m * 9.81 + Cl * vx * vx;
  const double Nf = FN_net * (lr / (lf + lr));
  const double Nr = FN_net * (lf / (lf + lr));

  const double alpha_f = delta - std::atan2(vy + r * lf, vx_s);
  const double alpha_r = -std::atan2(vy - r * lr, vx_s);

  const double Fy_f = Nf * D_ * std::sin(C_ * std::atan2(B * alpha_f, 1.0));
  const double Fy_r = Nr * D_ * std::sin(C_ * std::atan2(B * alpha_r, 1.0));

  const double F_fric = Cr0 + Cd * vx * vx;
  const double Fx     = Cm * T;

  dx[IX_EY]       = vx * std::sin(epsi) + vy * std::cos(epsi);
  dx[IX_EPSI]     = r - kappa * sdot_roll;
  dx[IX_VX]       = (1.0 / m) * (Fx * (1.0 + std::cos(delta)) - Fy_f * std::sin(delta) + m * vy * r - F_fric);
  dx[IX_VY]       = (1.0 / m) * (Fy_r + Fx * std::sin(delta) + Fy_f * std::cos(delta) - m * vx * r);
  dx[IX_R]        = (1.0 / Iz) * ((Fx * std::sin(delta) + Fy_f * std::cos(delta)) * lf - Fy_r * lr + Mtv);
  dx[IX_DELTA]    = ddelta;
  dx[IX_DDELTA]   = -(omega_n * omega_n) * delta - 2.0 * zeta * omega_n * ddelta + (omega_n * omega_n) * delta_cmd;
  dx[IX_DELTACMD] = u_ddelta_cmd;
  dx[IX_T]        = dT;
}

// ============================================================
// Solver-consistent discrete rollout helper
//
// Matches Python generator:
//
// x_mid  = x + 0.5*dt*f(x,u,kappa)
// x_next = x + dt*f(x_mid,u,kappa)
// s_next = s + dt*s_dot(x_mid,kappa)
//
// kappa is frozen on stage.
// ============================================================

inline void disc_step_midpoint_solver_consistent_(
    const std::array<double, NX>& x,
    const std::array<double, NU>& u,
    double kappa,
    double dt,
    const ParamBank& param,
    std::array<double, NX>& x_next,
    double& ds_next)
{
  std::array<double, NX> xdot{};
  std::array<double, NX> x_mid{};
  std::array<double, NX> xdot_mid{};

  // f(x_k, u_k, kappa_k)
  f_ode_frenet_(x, u, kappa, param, xdot);

  // x_mid = x_k + 0.5 * dt * f(x_k, ...)
  for (int i = 0; i < NX; ++i) {
    x_mid[i] = x[i] + 0.5 * dt * xdot[i];
  }

  // f(x_mid, u_k, kappa_k)
  f_ode_frenet_(x_mid, u, kappa, param, xdot_mid);

  // x_{k+1} = x_k + dt * f(x_mid, ...)
  for (int i = 0; i < NX; ++i) {
    x_next[i] = x[i] + dt * xdot_mid[i];
  }

  // s_{k+1} = s_k + dt * s_dot(x_mid, kappa_k)
  ds_next = dt * sdot_from_state_(x_mid, kappa);
}

// ============================================================
// Optional legacy RK4 helper (kept only for reference / debug)
// Not used anymore in rollout to keep consistency with solver.
// ============================================================

inline std::array<double, NX> rk4_step_frenet_(
    const std::array<double, NX>& x,
    const std::array<double, NU>& u,
    double kappa,
    double dt,
    const ParamBank& param)
{
  std::array<double, NX> k1{}, k2{}, k3{}, k4{};
  std::array<double, NX> xt{}, out{};

  f_ode_frenet_(x, u, kappa, param, k1);

  for (int i = 0; i < NX; ++i) xt[i] = x[i] + 0.5 * dt * k1[i];
  f_ode_frenet_(xt, u, kappa, param, k2);

  for (int i = 0; i < NX; ++i) xt[i] = x[i] + 0.5 * dt * k2[i];
  f_ode_frenet_(xt, u, kappa, param, k3);

  for (int i = 0; i < NX; ++i) xt[i] = x[i] + dt * k3[i];
  f_ode_frenet_(xt, u, kappa, param, k4);

  for (int i = 0; i < NX; ++i) {
    out[i] = x[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }

  return out;
}

// ============================================================
// Debug helpers
// ============================================================

static inline void dump_solver_status_(
    frenet_centerline_runtime_solver_capsule* capsule,
    int last_status)
{
  int qp_status = 0;
  int sqp_iter = 0;
  double res_stat = 0.0;
  double res_eq = 0.0;
  double res_ineq = 0.0;
  double res_comp = 0.0;

  ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
  ocp_nlp_get(capsule->nlp_solver, "sqp_iter", &sqp_iter);
  ocp_nlp_get(capsule->nlp_solver, "res_stat", &res_stat);
  ocp_nlp_get(capsule->nlp_solver, "res_eq", &res_eq);
  ocp_nlp_get(capsule->nlp_solver, "res_ineq", &res_ineq);
  ocp_nlp_get(capsule->nlp_solver, "res_comp", &res_comp);

  std::cout << std::fixed << std::setprecision(6)
            << "[MPC][FAIL] status=" << last_status
            << " qp_status=" << qp_status
            << " sqp_iter=" << sqp_iter
            << " res_stat=" << res_stat
            << " res_eq=" << res_eq
            << " res_ineq=" << res_ineq
            << " res_comp=" << res_comp
            << std::endl;
}

static inline void dump_seed_preview_(
    const std::vector<std::array<double, NX>>& x_stored,
    const std::vector<std::array<double, NU>>& u_stored,
    const std::vector<double>& s_rollout,
    const TrackSpline2D& track,
    const VelocityPlannerResult& vp)
{
  const int K = std::min(5, static_cast<int>(u_stored.size()));

  for (int k = 0; k < K; ++k) {
    const double s = s_rollout[k];
    const double kappa = track.getCurvature(s);
    const double vref = vp.valid ? vp.v_on_spline_interpolate(s) : x_stored[k][IX_VX];
    const double sdot_solver = sdot_from_state_(x_stored[k], kappa);

    std::cout << std::fixed << std::setprecision(4)
              << "[MPC][SEED k=" << k << "] "
              << "s=" << s
              << " ey=" << x_stored[k][IX_EY]
              << " epsi=" << x_stored[k][IX_EPSI]
              << " vx=" << x_stored[k][IX_VX]
              << " vy=" << x_stored[k][IX_VY]
              << " r=" << x_stored[k][IX_R]
              << " delta=" << x_stored[k][IX_DELTA]
              << " ddelta=" << x_stored[k][IX_DDELTA]
              << " delta_request=" << x_stored[k][IX_DELTACMD]
              << " T=" << x_stored[k][IX_T]
              << " | u_ddelta_cmd=" << u_stored[k][IU_DDELTACMD]
              << " dT=" << u_stored[k][IU_DT]
              << " Mtv=" << u_stored[k][IU_MTV]
              << " | kappa=" << kappa
              << " sdot=" << sdot_solver
              << " vref=" << vref
              << std::endl;
  }
}

static inline void dump_stage0_debug_(
    const std::array<double, NX>& x,
    const std::array<double, NU>& u,
    double s0,
    const VelocityPlannerResult& vp,
    const TrackSpline2D& track,
    const ParamBank& param)
{
  const double ey            = x[IX_EY];
  const double epsi          = x[IX_EPSI];
  const double vx            = x[IX_VX];
  const double vy            = x[IX_VY];
  const double r             = x[IX_R];
  const double delta         = x[IX_DELTA];
  const double ddelta        = x[IX_DDELTA];
  const double delta_request = x[IX_DELTACMD];
  const double T             = x[IX_T];

  const double u_ddelta_cmd = u[IU_DDELTACMD];
  const double dT           = u[IU_DT];
  const double Mtv          = u[IU_MTV];

  const double kappa = track.getCurvature(s0);
  const double vref  = vp.valid ? vp.v_on_spline_interpolate(s0) : vx;

  const double lf  = param.get("model_lf");
  const double lr  = param.get("model_lr");
  const double m   = param.get("model_m");
  const double B   = param.get("model_B");
  const double C_  = param.get("model_C");
  const double D_  = param.get("model_D");
  const double Cr0 = param.get("model_Cr0");
  const double Cd  = param.get("model_Cd");
  const double Cl  = param.get("model_Cl");
  const double Cm  = param.get("model_Cm");

  const double mux = get_mux_eff_(param);
  const double muy = get_muy_eff_(param);

  const double L_c = param.get("mpc_constraints_L_c");
  const double W_c = param.get("mpc_constraints_W_c");
  const double track_width = param.get("mpc_constraints_track_width");
  const double track_half_width = 0.5 * track_width;

  const double vx_s = std::sqrt(vx * vx + 1e-8);
  const double sdot_solver = (vx * std::cos(epsi) - vy * std::sin(epsi)) / (1.0 - ey * kappa);

  const double beta_dyn = std::atan2(vy, vx_s);
  const double beta_kin = std::atan2(lr * std::tan(delta), (lf + lr));
  const double beta_err = beta_dyn - beta_kin;

  const double FN_net = m * 9.81 + Cl * vx * vx;
  const double Nf = FN_net * (lr / (lf + lr));
  const double Nr = FN_net * (lf / (lf + lr));

  const double alpha_f = delta - std::atan2(vy + r * lf, vx_s);
  const double alpha_r = -std::atan2(vy - r * lr, vx_s);

  const double Fy_f = Nf * D_ * std::sin(C_ * std::atan2(B * alpha_f, 1.0));
  const double Fy_r = Nr * D_ * std::sin(C_ * std::atan2(B * alpha_r, 1.0));
  const double Fx   = Cm * T;
  const double Ffr  = Cr0 + Cd * vx * vx;

  const double epsi_abs = std::sqrt(epsi * epsi + 1e-8);
  const double track_left  =  ey + L_c * std::sin(epsi_abs) + W_c * std::cos(epsi) - track_half_width;
  const double track_right = -ey + L_c * std::sin(epsi_abs) + W_c * std::cos(epsi) - track_half_width;

  const double Nf_safe = std::sqrt(Nf * Nf + 1.0);
  const double Nr_safe = std::sqrt(Nr * Nr + 1.0);

  const double h_fric_f = std::pow(Fx / (mux * Nf_safe), 2) + std::pow(Fy_f / (muy * Nf_safe), 2) - 1.0;
  const double h_fric_r = std::pow(Fx / (mux * Nr_safe), 2) + std::pow(Fy_r / (muy * Nr_safe), 2) - 1.0;

  std::cout << std::fixed << std::setprecision(6)
            << "[MPC][STAGE0] "
            << "s=" << s0
            << " ey=" << ey
            << " epsi=" << epsi
            << " vx=" << vx
            << " vy=" << vy
            << " r=" << r
            << " delta=" << delta
            << " delta_dot=" << ddelta
            << " delta_request=" << delta_request
            << " T=" << T
            << " | u_ddelta_cmd=" << u_ddelta_cmd
            << " dT=" << dT
            << " Mtv=" << Mtv
            << " | kappa=" << kappa
            << " sdot=" << sdot_solver
            << " vref=" << vref
            << " | beta_dyn=" << beta_dyn
            << " beta_kin=" << beta_kin
            << " beta_err=" << beta_err
            << " | af=" << alpha_f
            << " ar=" << alpha_r
            << " Fx=" << Fx
            << " Fy_f=" << Fy_f
            << " Fy_r=" << Fy_r
            << " Ffr=" << Ffr
            << " | trL=" << track_left
            << " trR=" << track_right
            << " hf=" << h_fric_f
            << " hr=" << h_fric_r
            << std::endl;
}

static inline void dump_runtime_params_stage0_(
    const ParamBank& param,
    const VelocityPlannerResult& vp,
    const std::vector<double>& s_rollout,
    const std::vector<std::array<double, NX>>& x_stored,
    const TrackSpline2D& track)
{
  if (s_rollout.empty() || x_stored.empty()) return;

  const double s0 = s_rollout[0];
  const double kappa0 = track.getCurvature(s0);
  const double vref0 = vp.valid ? vp.v_on_spline_interpolate(s0) : x_stored[0][IX_VX];

  std::cout << std::fixed << std::setprecision(4)
            << "[MPC][P0] "
            << "kappa0=" << kappa0
            << " vref0=" << vref0
            << " q_ey=" << get_q_ey_(param)
            << " gamma_progress(q_sdot)=" << get_gamma_progress_(param)
            << " Q_epsi=" << get_q_epsi_(param)
            << " Q_beta_dyn_kin=" << get_q_beta_dyn_kin_(param)
            << " R_u_ddelta_cmd=" << get_r_u_ddelta_cmd_(param)
            << " R_dT=" << get_r_dT_(param)
            << " R_Mtv=" << get_r_Mtv_(param)
            << " q_ey_terminal=" << get_q_ey_terminal_(param)
            << " Q_epsi_terminal=" << get_q_epsi_terminal_(param)
            << " mux_eff=" << get_mux_eff_(param)
            << " muy_eff=" << get_muy_eff_(param)
            << std::endl;

  std::cout << std::fixed << std::setprecision(4)
            << "[MPC][BOUNDS] "
            << "ey=[" << param.get("mpc_bounds_min_ey") << "," << param.get("mpc_bounds_max_ey") << "] "
            << "epsi=[" << param.get("mpc_bounds_min_epsi") << "," << param.get("mpc_bounds_max_epsi") << "] "
            << "vx=[" << param.get("mpc_bounds_min_vx") << "," << param.get("mpc_bounds_max_vx") << "] "
            << "vy=[" << param.get("mpc_bounds_min_vy") << "," << param.get("mpc_bounds_max_vy") << "] "
            << "r=[" << param.get("mpc_bounds_min_r") << "," << param.get("mpc_bounds_max_r") << "] "
            << "delta=[" << param.get("mpc_bounds_min_delta") << "," << param.get("mpc_bounds_max_delta") << "] "
            << "delta_dot=[" << get_min_state_ddelta_(param) << "," << get_max_state_ddelta_(param) << "] "
            << "delta_request=[" << get_min_delta_cmd_state_(param) << "," << get_max_delta_cmd_state_(param) << "] "
            << "T=[" << param.get("mpc_bounds_min_T") << "," << param.get("mpc_bounds_max_T") << "] "
            << "u_ddelta_cmd=[" << get_min_u_ddelta_cmd_(param) << "," << get_max_u_ddelta_cmd_(param) << "] "
            << "dT=[" << param.get("mpc_bounds_min_dT") << "," << param.get("mpc_bounds_max_dT") << "] "
            << "Mtv=[" << param.get("mpc_bounds_min_Mtv") << "," << param.get("mpc_bounds_max_Mtv") << "]"
            << std::endl;
}

} // namespace

// ============================================================
// Lifecycle
// ============================================================

MPCCInterface::MPCCInterface(const ParamBank& P) : param_(P) { init_(); }
MPCCInterface::~MPCCInterface() { destroy_(); }

void MPCCInterface::setParams(const ParamBank& P) { param_ = P; }

void MPCCInterface::setTrack(const TrackSpline2D& track) {
  track_ = track;
  has_track_ = true;
}

void MPCCInterface::requestInitialGuessReset() { initialized_ = false; }

// ============================================================
// init / destroy
// ============================================================

void MPCCInterface::init_() {
  if (is_initialized_) return;

  static_assert(NP_EXPECTED == NP, "Mismatch between Python p layout and generated NP");
  static_assert(NX == 9, "This wrapper expects Frenet solver with NX=9");
  static_assert(NU == 3, "This wrapper expects Frenet solver with NU=3");
  static_assert(NY == 7, "This wrapper expects CONVEX_OVER_NONLINEAR stage NY = 7");
  static_assert(NYN == 2, "This wrapper expects CONVEX_OVER_NONLINEAR terminal NYN = 2");

  capsule_ = frenet_centerline_runtime_acados_create_capsule();
  if (!capsule_) {
    throw std::runtime_error("MPCCInterface: capsule alloc failed");
  }

  if (frenet_centerline_runtime_acados_create(capsule_) != 0) {
    throw std::runtime_error("MPCCInterface: acados_create failed");
  }

  x_stored_.resize(N + 1);
  u_stored_.resize(N);
  s_rollout_.assign(N + 1, 0.0);

  for (auto& x : x_stored_) x.fill(0.0);
  for (auto& u : u_stored_) u.fill(0.0);

  n_rti_iterations_    = static_cast<int>(param_.get("mpc_solver_n_sqp"));
  n_reset_threshold_   = static_cast<int>(param_.get("mpc_solver_n_reset"));
  n_consecutive_fails_ = 0;

  is_initialized_ = true;
}

void MPCCInterface::destroy_() {
  if (!is_initialized_) return;
  frenet_centerline_runtime_acados_free(capsule_);
  frenet_centerline_runtime_acados_free_capsule(capsule_);
  capsule_ = nullptr;
  is_initialized_ = false;
}

// ============================================================
// Helpers
// ============================================================

double MPCCInterface::dt_() const {
  return param_.get("mpc_dt");
}

double MPCCInterface::wrap_mod_s_(double s) const {
  const double L = track_.totalLength();
  if (L <= 0.0) return s;
  double r = std::fmod(s, L);
  if (r < 0.0) r += L;
  return r;
}

double MPCCInterface::unwrapAngle_(double prev, double curr) {
  double d = curr - prev;
  while (d > M_PI) { curr -= 2.0 * M_PI; d = curr - prev; }
  while (d < -M_PI) { curr += 2.0 * M_PI; d = curr - prev; }
  return curr;
}

// ============================================================
// Runtime slack costs
// ============================================================

void MPCCInterface::apply_runtime_slack_costs_() {
  const double q_track_lin  = param_.get("mpc_cost_q_slack_track_lin");
  const double q_track_quad = param_.get("mpc_cost_q_slack_track_quad");
  const double q_fric_lin   = param_.get("mpc_cost_q_slack_fric_lin");
  const double q_fric_quad  = param_.get("mpc_cost_q_slack_fric_quad");

  double zl[4] = {q_track_lin, q_track_lin, q_fric_lin, q_fric_lin};
  double zu[4] = {q_track_lin, q_track_lin, q_fric_lin, q_fric_lin};
  double Zl[4] = {q_track_quad, q_track_quad, q_fric_quad, q_fric_quad};
  double Zu[4] = {q_track_quad, q_track_quad, q_fric_quad, q_fric_quad};

  for (int k = 0; k < N; ++k) {
    ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, k, "zl", zl);
    ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, k, "zu", zu);
    ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, k, "Zl", Zl);
    ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, k, "Zu", Zu);
  }

  ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, N, "zl", zl);
  ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, N, "zu", zu);
  ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, N, "Zl", Zl);
  ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, N, "Zu", Zu);
}

// ============================================================
// Runtime box bounds
// ============================================================

void MPCCInterface::apply_runtime_box_bounds_() {
  double lbx[NX] = {
      param_.get("mpc_bounds_min_ey"),
      param_.get("mpc_bounds_min_epsi"),
      param_.get("mpc_bounds_min_vx"),
      param_.get("mpc_bounds_min_vy"),
      param_.get("mpc_bounds_min_r"),
      param_.get("mpc_bounds_min_delta"),
      get_min_state_ddelta_(param_),
      get_min_delta_cmd_state_(param_),
      param_.get("mpc_bounds_min_T")
  };

  double ubx[NX] = {
      param_.get("mpc_bounds_max_ey"),
      param_.get("mpc_bounds_max_epsi"),
      param_.get("mpc_bounds_max_vx"),
      param_.get("mpc_bounds_max_vy"),
      param_.get("mpc_bounds_max_r"),
      param_.get("mpc_bounds_max_delta"),
      get_max_state_ddelta_(param_),
      get_max_delta_cmd_state_(param_),
      param_.get("mpc_bounds_max_T")
  };

  double lbu[NU] = {
      get_min_u_ddelta_cmd_(param_),
      param_.get("mpc_bounds_min_dT"),
      param_.get("mpc_bounds_min_Mtv")
  };

  double ubu[NU] = {
      get_max_u_ddelta_cmd_(param_),
      param_.get("mpc_bounds_max_dT"),
      param_.get("mpc_bounds_max_Mtv")
  };

  for (int k = 0; k < N; ++k) {
    ocp_nlp_constraints_model_set(
        capsule_->nlp_config, capsule_->nlp_dims,
        capsule_->nlp_in, capsule_->nlp_out, k, "lbu", lbu);

    ocp_nlp_constraints_model_set(
        capsule_->nlp_config, capsule_->nlp_dims,
        capsule_->nlp_in, capsule_->nlp_out, k, "ubu", ubu);
  }

  for (int k = 1; k < N; ++k) {
    ocp_nlp_constraints_model_set(
        capsule_->nlp_config, capsule_->nlp_dims,
        capsule_->nlp_in, capsule_->nlp_out, k, "lbx", lbx);

    ocp_nlp_constraints_model_set(
        capsule_->nlp_config, capsule_->nlp_dims,
        capsule_->nlp_in, capsule_->nlp_out, k, "ubx", ubx);
  }

  ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims,
      capsule_->nlp_in, capsule_->nlp_out, N, "lbx", lbx);

  ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims,
      capsule_->nlp_in, capsule_->nlp_out, N, "ubx", ubx);
}

void MPCCInterface::unwrap_initial_guess_() {}

// ============================================================
// x0 hard constraint
// ============================================================

void MPCCInterface::set_x0_hard_(const std::array<double, NX>& x0_in) {
  std::array<double, NX> x0 = clamp_state_to_runtime_bounds_(x0_in, param_);

  ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims,
      capsule_->nlp_in, capsule_->nlp_out, 0, "lbx",
      const_cast<double*>(x0.data()));

  ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims,
      capsule_->nlp_in, capsule_->nlp_out, 0, "ubx",
      const_cast<double*>(x0.data()));

  ocp_nlp_out_set(
      capsule_->nlp_config, capsule_->nlp_dims,
      capsule_->nlp_out, capsule_->nlp_in, 0, "x",
      const_cast<double*>(x0.data()));
}

// ============================================================
// Cold start
// ============================================================

void MPCCInterface::coldstart_guess_(const std::array<double, NX>& x0_in) {
  const double dt = dt_();

  std::array<double, NX> x0 = clamp_state_to_runtime_bounds_(x0_in, param_);

  s_rollout_[0] = s0_global_wrapped_;
  x_stored_[0]  = x0;

  ocp_nlp_out_set(capsule_->nlp_config, capsule_->nlp_dims,
                  capsule_->nlp_out, capsule_->nlp_in, 0, "x", x_stored_[0].data());

  for (int k = 0; k < N; ++k) {
    u_stored_[k] = make_zero_u_();

    ocp_nlp_out_set(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, capsule_->nlp_in, k, "u", u_stored_[k].data());

    const double kappa_k = track_.getCurvature(s_rollout_[k]);

    std::array<double, NX> xk1{};
    double ds_k = 0.0;

    disc_step_midpoint_solver_consistent_(
        x_stored_[k], u_stored_[k], kappa_k, dt, param_, xk1, ds_k);

    xk1 = clamp_state_to_runtime_bounds_(xk1, param_);

    x_stored_[k + 1]  = xk1;
    s_rollout_[k + 1] = wrap_mod_s_(s_rollout_[k] + ds_k);

    ocp_nlp_out_set(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, capsule_->nlp_in, k + 1, "x", x_stored_[k + 1].data());
  }
}

// ============================================================
// Warm start
// ============================================================

void MPCCInterface::warmstart_shift_(const std::array<double, NX>& x0_in) {
  const double dt = dt_();

  std::array<double, NX> x0 = clamp_state_to_runtime_bounds_(x0_in, param_);

  if (N >= 2) {
    for (int k = 0; k < N - 1; ++k) {
      u_stored_[k] = u_stored_[k + 1];
    }
    u_stored_[N - 1] = u_stored_[N - 2];
  }

  x_stored_[0]  = x0;
  s_rollout_[0] = s0_global_wrapped_;

  for (int k = 0; k < N; ++k) {
    const double kappa_k = track_.getCurvature(s_rollout_[k]);

    const std::array<double, NX> xk = x_stored_[k];
    std::array<double, NX> xk1{};
    double ds_k = 0.0;

    disc_step_midpoint_solver_consistent_(
        xk, u_stored_[k], kappa_k, dt, param_, xk1, ds_k);

    xk1 = clamp_state_to_runtime_bounds_(xk1, param_);

    x_stored_[k + 1]  = xk1;
    s_rollout_[k + 1] = wrap_mod_s_(s_rollout_[k] + ds_k);
  }

  for (int k = 0; k <= N; ++k) {
    ocp_nlp_out_set(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, capsule_->nlp_in, k, "x", x_stored_[k].data());
  }

  for (int k = 0; k < N; ++k) {
    ocp_nlp_out_set(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, capsule_->nlp_in, k, "u", u_stored_[k].data());
  }
}

// ============================================================
// Rebuild s rollout
// ============================================================

void MPCCInterface::rebuild_s_rollout_from_solution_() {
  const double dt = dt_();
  s_rollout_[0] = s0_global_wrapped_;

  for (int k = 0; k < N; ++k) {
    const double kappa_k = track_.getCurvature(s_rollout_[k]);

    std::array<double, NX> x_dummy{};
    double ds_k = 0.0;

    disc_step_midpoint_solver_consistent_(
        x_stored_[k], u_stored_[k], kappa_k, dt, param_, x_dummy, ds_k);

    s_rollout_[k + 1] = wrap_mod_s_(s_rollout_[k] + ds_k);
  }
}

// ============================================================
// Stage params
//
// p = [
//   kappa, v_ref, mux, muy,
//   q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress,
//   q_ey_terminal, q_epsi_terminal
// ]
// ============================================================

void MPCCInterface::set_stage_params_(const VelocityPlannerResult& vp) {
  constexpr int np_stage = FRENET_CENTERLINE_RUNTIME_NP;
  static_assert(np_stage == NP, "NP mismatch: generated solver vs C++ constants");

  const double mux_eff         = get_mux_eff_(param_);
  const double muy_eff         = get_muy_eff_(param_);

  const double q_ey            = get_q_ey_(param_);
  const double q_epsi          = get_q_epsi_(param_);
  const double r_u_ddelta_cmd  = get_r_u_ddelta_cmd_(param_);
  const double r_dT            = get_r_dT_(param_);
  const double r_Mtv           = get_r_Mtv_(param_);
  const double q_beta_dyn_kin  = get_q_beta_dyn_kin_(param_);
  const double gamma_progress  = get_gamma_progress_(param_);
  const double q_ey_terminal   = get_q_ey_terminal_(param_);
  const double q_epsi_terminal = get_q_epsi_terminal_(param_);

  for (int k = 0; k <= N; ++k) {
    double p[NP];
    std::memset(p, 0, sizeof(p));

    const double s_k = s_rollout_[k];
    const double kappa_k = track_.getCurvature(s_k);
    const double vx_fallback = x_stored_[std::min(k, N)][IX_VX];
    const double vref_k = planner_vref_at_s_(vp, s_k, vx_fallback, param_);

    p[IDX_KAPPA]           = kappa_k;
    p[IDX_VREF]            = vref_k;
    p[IDX_MUX]             = mux_eff;
    p[IDX_MUY]             = muy_eff;

    p[IDX_Q_EY]            = q_ey;
    p[IDX_Q_EPSI]          = q_epsi;
    p[IDX_R_U_DDELTACMD]   = r_u_ddelta_cmd;
    p[IDX_R_DT]            = r_dT;
    p[IDX_R_MTV]           = r_Mtv;
    p[IDX_Q_BETA_DYN_KIN]  = q_beta_dyn_kin;
    p[IDX_GAMMA_PROGRESS]  = gamma_progress;

    // Zostawiam jak było poza rolloutem — świadomie nie ruszam kosztu.
    p[IDX_Q_EY_TERMINAL]   = q_ey;
    p[IDX_Q_EPSI_TERMINAL] = q_epsi;

    frenet_centerline_runtime_acados_update_params(capsule_, k, p, np_stage);
  }

  last_cx_.resize(N + 1);
  last_cy_.resize(N + 1);
  for (int k = 0; k <= N; ++k) {
    last_cx_[k] = track_.getX(s_rollout_[k]);
    last_cy_[k] = track_.getY(s_rollout_[k]);
  }
}

// ============================================================
// CONVEX_OVER_NONLINEAR residual references
// stage y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_err, s_dot]
// term  y = [ey, epsi]
// We keep them all at zero.
// ============================================================

void MPCCInterface::set_stage_yref_from_planner_(const VelocityPlannerResult& /*vp*/) {
  double yref[NY];
  double yref_e[NYN];

  for (int i = 0; i < NY; ++i)  yref[i] = 0.0;
  for (int i = 0; i < NYN; ++i) yref_e[i] = 0.0;

  for (int k = 0; k < N; ++k) {
    ocp_nlp_cost_model_set(
        capsule_->nlp_config,
        capsule_->nlp_dims,
        capsule_->nlp_in,
        k,
        "yref",
        yref);
  }

  ocp_nlp_cost_model_set(
      capsule_->nlp_config,
      capsule_->nlp_dims,
      capsule_->nlp_in,
      N,
      "yref",
      yref_e);
}

void MPCCInterface::set_zero_yref_() {
  double yref[NY];
  for (int i = 0; i < NY; ++i) yref[i] = 0.0;

  for (int k = 0; k < N; ++k) {
    ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims,
                           capsule_->nlp_in, k, "yref", yref);
  }

  double yref_e[NYN];
  for (int i = 0; i < NYN; ++i) yref_e[i] = 0.0;

  ocp_nlp_cost_model_set(capsule_->nlp_config, capsule_->nlp_dims,
                         capsule_->nlp_in, N, "yref", yref_e);
}

// ============================================================
// Solve
// ============================================================

MPCC_Return MPCCInterface::solve(const MPCC_State& x0_in, double theta0_wrapped) {
  if (!has_track_) {
    throw std::runtime_error("MPCCInterface: track not set");
  }
  if (!is_initialized_) {
    init_();
  }

  const bool hard_reset_before_every_solve =
      param_.get("mpc_solver_hard_reset_before_every_solve") > 0.5;

  if (hard_reset_before_every_solve) {
    std::cout << "[MPC][RESET] hard reset solver capsule before solve" << std::endl;

    destroy_();
    init_();

    initialized_ = false;
    n_consecutive_fails_ = 0;
    s_rollout_.assign(N + 1, 0.0);
    last_cx_.clear();
    last_cy_.clear();
  }

  ++g_solve_call_counter;

  s0_global_wrapped_ = wrap_mod_s_(theta0_wrapped);

  std::array<double, NX> x0 =
      measured_cartesian_to_frenet_(x0_in, track_, s0_global_wrapped_);

  x0 = clamp_state_to_runtime_bounds_(x0, param_);

  if (g_solve_call_counter <= DEBUG_PREVIEW_FIRST_CALLS) {
    std::cout << std::fixed << std::setprecision(4)
              << "[MPC][X0] "
              << "X=" << x0_in.X
              << " Y=" << x0_in.Y
              << " phi=" << x0_in.phi
              << " vx=" << x0_in.vx
              << " vy=" << x0_in.vy
              << " r=" << x0_in.r
              << " theta_global=" << theta0_wrapped
              << " delta=" << x0_in.delta
              << " delta_dot=" << x0_in.delta_dot
              << " delta_request=" << x0_in.delta_request
              << " T=" << x0_in.T
              << " | frenet ey=" << x0[IX_EY]
              << " epsi=" << x0[IX_EPSI]
              << " vx=" << x0[IX_VX]
              << " vy=" << x0[IX_VY]
              << " r=" << x0[IX_R]
              << " delta=" << x0[IX_DELTA]
              << " delta_dot=" << x0[IX_DDELTA]
              << " delta_request=" << x0[IX_DELTACMD]
              << " T=" << x0[IX_T]
              << std::endl;
  }

  if (!initialized_) {
    coldstart_guess_(x0);
    set_x0_hard_(x0);
  } else {
    warmstart_shift_(x0);
    set_x0_hard_(x0);
  }

  apply_runtime_box_bounds_();
  apply_runtime_slack_costs_();

  State bolide;
  bolide.X        = x0_in.X;
  bolide.Y        = x0_in.Y;
  bolide.yaw      = x0_in.phi;
  bolide.delta    = x0_in.delta;
  bolide.vx       = x0_in.vx;
  bolide.vy       = x0_in.vy;
  bolide.yaw_rate = x0_in.r;
  bolide.acc      = 0.0;

  VelocityPlannerResult vp =
      velocity_planner_process_for_control(param_, track_, bolide, s0_global_wrapped_);

  set_stage_params_(vp);
  set_stage_yref_from_planner_(vp);

  if (g_solve_call_counter <= DEBUG_PREVIEW_FIRST_CALLS) {
    dump_runtime_params_stage0_(param_, vp, s_rollout_, x_stored_, track_);
    dump_seed_preview_(x_stored_, u_stored_, s_rollout_, track_, vp);
    dump_stage0_debug_(x_stored_[0], u_stored_[0], s_rollout_[0], vp, track_, param_);
  }

  const int solver_status = frenet_centerline_runtime_acados_solve(capsule_);

  MPCC_Return out{};
  out.success = (solver_status == 0 || solver_status == 2);

  if (out.success) {
    initialized_ = true;

    for (int k = 0; k <= N; ++k) {
      ocp_nlp_out_get(capsule_->nlp_config, capsule_->nlp_dims,
                      capsule_->nlp_out, k, "x", x_stored_[k].data());
    }
    for (int k = 0; k < N; ++k) {
      ocp_nlp_out_get(capsule_->nlp_config, capsule_->nlp_dims,
                      capsule_->nlp_out, k, "u", u_stored_[k].data());
    }

    rebuild_s_rollout_from_solution_();

    out.ddelta_request = u_stored_[0][IU_DDELTACMD];
    out.dT             = u_stored_[0][IU_DT];
    out.Mtv            = u_stored_[0][IU_MTV];

    const double sdot_1 =
        sdot_from_state_(x_stored_[1], track_.getCurvature(s_rollout_[1]));

    const double vref_next =
        planner_vref_at_s_(vp, s_rollout_[1], x_stored_[1][IX_VX], param_);

    out.next_vtheta        = sdot_1;
    out.next_vref          = vref_next;
    out.next_yaw_rate      = x_stored_[1][IX_R];
    out.next_vx_target     = x_stored_[1][IX_VX];
    out.next_vy_target     = x_stored_[1][IX_VY];

    out.next_delta         = x_stored_[1][IX_DELTA];
    out.next_delta_dot     = x_stored_[1][IX_DDELTA];
    out.next_delta_request = x_stored_[1][IX_DELTACMD];
    out.next_T             = x_stored_[1][IX_T];

    const double dt = dt_();
    const double dvx_dt = (x_stored_[1][IX_VX] - x0[IX_VX]) / std::max(1e-6, dt);
    const double dvy_dt = (x_stored_[1][IX_VY] - x0[IX_VY]) / std::max(1e-6, dt);

    out.ax = dvx_dt; // kept as in original
    out.ay = dvy_dt; // kept as in original

    out.X_mpc.resize(N + 1);
    out.Y_mpc.resize(N + 1);

    for (int k = 0; k <= N; ++k) {
      double Xk = 0.0;
      double Yk = 0.0;
      frenet_to_cartesian_(track_, s_rollout_[k], x_stored_[k][IX_EY], Xk, Yk);
      out.X_mpc(k) = Xk;
      out.Y_mpc(k) = Yk;
    }

    last_cx_.resize(N + 1);
    last_cy_.resize(N + 1);
    for (int k = 0; k <= N; ++k) {
      last_cx_[k] = track_.getX(s_rollout_[k]);
      last_cy_[k] = track_.getY(s_rollout_[k]);
    }

    n_consecutive_fails_ = 0;
    return out;
  }

  {
    std::array<double, NX> x_dbg{};
    std::array<double, NU> u_dbg{};

    ocp_nlp_out_get(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, 0, "x", x_dbg.data());
    ocp_nlp_out_get(capsule_->nlp_config, capsule_->nlp_dims,
                    capsule_->nlp_out, 0, "u", u_dbg.data());

    dump_solver_status_(capsule_, solver_status);
    dump_runtime_params_stage0_(param_, vp, s_rollout_, x_stored_, track_);
    dump_seed_preview_(x_stored_, u_stored_, s_rollout_, track_, vp);
    dump_stage0_debug_(x_dbg, u_dbg, s_rollout_[0], vp, track_, param_);

    const bool hard_reset_after_fail =
        param_.get("mpc_solver_hard_reset_after_fail") > 0.5;

    if (hard_reset_after_fail) {
      std::cout << "[MPC][RESET] hard reset solver capsule after failed solve" << std::endl;

      destroy_();
      init_();

      initialized_ = false;
      n_consecutive_fails_ = 0;
      s_rollout_.assign(N + 1, 0.0);
      last_cx_.clear();
      last_cy_.clear();

      return out;
    }

    n_consecutive_fails_++;
    if (n_consecutive_fails_ >= n_reset_threshold_) {
      initialized_ = false;
      n_consecutive_fails_ = 0;
    }
  }

  return out;
}

// ============================================================
// Getter
// ============================================================

void MPCCInterface::getLastSampledPath(std::vector<double>& cx, std::vector<double>& cy) const {
  cx = last_cx_;
  cy = last_cy_;
}

} // namespace v2_control
