#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "ParamBank.hpp"
#include "spline.hpp"

namespace v2_control {
struct VelocityPlannerResult;
}

extern "C" {
#include "acados_solver_frenet_centerline_runtime.h"
#include "acados_c/ocp_nlp_interface.h"
}

namespace v2_control {

// ============================================================
// Solver dimensions
// Must match generated Frenet solver exactly
// ============================================================

constexpr int NX = FRENET_CENTERLINE_RUNTIME_NX;
constexpr int NU = FRENET_CENTERLINE_RUNTIME_NU;
constexpr int N  = FRENET_CENTERLINE_RUNTIME_N;
constexpr int NP = FRENET_CENTERLINE_RUNTIME_NP;

constexpr int NY  = FRENET_CENTERLINE_RUNTIME_NY;
constexpr int NYN = FRENET_CENTERLINE_RUNTIME_NYN;

// ============================================================
// Internal state layout
// x = [ey, epsi, vx, vy, r, delta, ddelta, delta_cmd, T]
// ============================================================

constexpr int IX_EY       = 0;
constexpr int IX_EPSI     = 1;
constexpr int IX_VX       = 2;
constexpr int IX_VY       = 3;
constexpr int IX_R        = 4;
constexpr int IX_DELTA    = 5;
constexpr int IX_DDELTA   = 6;
constexpr int IX_DELTACMD = 7;
constexpr int IX_T        = 8;

// ============================================================
// Input layout
// u = [u_ddelta_cmd, dT, Mtv]
// ============================================================

constexpr int IU_DDELTACMD = 0;
constexpr int IU_DT        = 1;
constexpr int IU_MTV       = 2;

// ============================================================
// Runtime parameter layout
//
// p = [
//   kappa, v_ref, mux, muy,
//   q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress,
//   q_ey_terminal, q_epsi_terminal
// ]
//
// NOTE:
// - v_ref is kept mainly for ABI compatibility / debug / outputs.
// - bounds are runtime via constraint setters in C++
// - slack costs are runtime via zl/zu/Zl/Zu setters in C++
// ============================================================

constexpr int IDX_KAPPA           = 0;
constexpr int IDX_VREF            = 1;
constexpr int IDX_MUX             = 2;
constexpr int IDX_MUY             = 3;

constexpr int IDX_Q_EY            = 4;
constexpr int IDX_Q_EPSI          = 5;
constexpr int IDX_R_U_DDELTACMD   = 6;
constexpr int IDX_R_DT            = 7;
constexpr int IDX_R_MTV           = 8;
constexpr int IDX_Q_BETA_DYN_KIN  = 9;
constexpr int IDX_GAMMA_PROGRESS  = 10;

constexpr int IDX_Q_EY_TERMINAL   = 11;
constexpr int IDX_Q_EPSI_TERMINAL = 12;

constexpr int NP_EXPECTED = 13;

// ============================================================
// CONVEX_OVER_NONLINEAR cost layout
//
// stage y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_dyn_minus_beta_kin, s_dot]
// term  y = [ey, epsi]
// ============================================================

constexpr int IY_EY                      = 0;
constexpr int IY_EPSI                    = 1;
constexpr int IY_U_DDELTACMD             = 2;
constexpr int IY_D_T                     = 3;
constexpr int IY_MTV                     = 4;
constexpr int IY_BETA_DYN_MINUS_BETA_KIN = 5;
constexpr int IY_S_DOT                   = 6;

constexpr int IYE_EY   = 0;
constexpr int IYE_EPSI = 1;

// ============================================================
// Compile-time sanity checks
// ============================================================

static_assert(NX == 9,  "Expected Frenet solver with NX == 9");
static_assert(NU == 3,  "Expected Frenet solver with NU == 3");
static_assert(NP == 13, "Expected Frenet solver with NP == 13");
static_assert(NP == NP_EXPECTED, "NP mismatch between header and generated solver");

static_assert(NY  == 7, "Expected CONVEX_OVER_NONLINEAR stage dim NY == 7");
static_assert(NYN == 2, "Expected CONVEX_OVER_NONLINEAR terminal dim NYN == 2");

static_assert(IY_EY == 0, "Unexpected IY_EY");
static_assert(IY_EPSI == 1, "Unexpected IY_EPSI");
static_assert(IY_U_DDELTACMD == 2, "Unexpected IY_U_DDELTACMD");
static_assert(IY_D_T == 3, "Unexpected IY_D_T");
static_assert(IY_MTV == 4, "Unexpected IY_MTV");
static_assert(IY_BETA_DYN_MINUS_BETA_KIN == 5, "Unexpected IY_BETA_DYN_MINUS_BETA_KIN");
static_assert(IY_S_DOT == 6, "Unexpected IY_S_DOT");

// ============================================================
// External wrapper-facing state
// Wrapper accepts cartesian/global state.
// Extra steering states are carried in the wrapper.
// ============================================================

struct MPCC_State {
  double X     = 0.0;
  double Y     = 0.0;
  double phi   = 0.0;

  double vx    = 0.0;
  double vy    = 0.0;
  double r     = 0.0;

  double theta = 0.0;

  double delta         = 0.0;
  double delta_dot     = 0.0;
  double delta_request = 0.0;

  double T = 0.0;
};

// ============================================================
// Solver output
// Real steering-related control is u_ddelta_cmd, exposed here
// as ddelta_request for continuity with the rest of the code.
// ============================================================

struct MPCC_Return {
  double ddelta_request = 0.0;
  double dT             = 0.0;
  double Mtv            = 0.0;

  bool success = false;

  double next_delta         = 0.0;
  double next_delta_dot     = 0.0;
  double next_delta_request = 0.0;

  double next_T = 0.0;

  double next_yaw_rate  = 0.0;
  double ax             = 0.0;
  double ay             = 0.0;
  double next_vx_target = 0.0;
  double next_vy_target = 0.0;

  double next_vtheta = 0.0;
  double next_vref   = 0.0;

  Eigen::VectorXd X_mpc;
  Eigen::VectorXd Y_mpc;
};

class MPCCInterface {
public:
  MPCCInterface() = default;
  explicit MPCCInterface(const ParamBank& P);
  ~MPCCInterface();

  MPCCInterface(const MPCCInterface&) = delete;
  MPCCInterface& operator=(const MPCCInterface&) = delete;

  void setParams(const ParamBank& P);
  void setTrack(const TrackSpline2D& track);
  void requestInitialGuessReset();

  MPCC_Return solve(const MPCC_State& x0_in, double theta0_wrapped);

  void getLastSampledPath(std::vector<double>& cx, std::vector<double>& cy) const;

private:
  void init_();
  void destroy_();

  double dt_() const;
  double wrap_mod_s_(double s_unwrapped) const;
  static double unwrapAngle_(double prev, double curr);

  // ==========================================================
  // Runtime numerics
  // ==========================================================
  void apply_runtime_box_bounds_();
  void apply_runtime_slack_costs_();

  void unwrap_initial_guess_();

  void set_x0_hard_(const std::array<double, NX>& x0);

  // Initial guess generation:
  // kept solver-consistent with generated DISCRETE midpoint dynamics:
  // x_{k+1} = x_k + dt * f(x_mid, u_k, kappa_k),
  // x_mid   = x_k + 0.5*dt*f(x_k, u_k, kappa_k)
  //
  // s_rollout is rebuilt consistently from midpoint s_dot using the same
  // frozen stage curvature kappa_k as the solver parameterization.
  void coldstart_guess_(const std::array<double, NX>& x0);
  void warmstart_shift_(const std::array<double, NX>& x0);
  void rebuild_s_rollout_from_solution_();

  // p = [
  //   kappa, v_ref, mux, muy,
  //   q_ey, q_epsi, r_u_ddelta_cmd, r_dT, r_Mtv, q_beta_dyn_kin, gamma_progress,
  //   q_ey_terminal, q_epsi_terminal
  // ]
  void set_stage_params_(const VelocityPlannerResult& vp);

  // CONVEX_OVER_NONLINEAR:
  // stage residual references are zero for
  // y = [ey, epsi, u_ddelta_cmd, dT, Mtv, beta_dyn_minus_beta_kin, s_dot]
  // terminal residual references are zero for
  // y_e = [ey, epsi]
  void set_stage_yref_from_planner_(const VelocityPlannerResult& vp);
  void set_zero_yref_();

private:
  ParamBank param_;

  // Global wrapped spline coordinate used as stage-0 Frenet anchor.
  double s0_global_wrapped_ = 0.0;

  bool is_initialized_ = false;
  frenet_centerline_runtime_solver_capsule* capsule_ = nullptr;

  TrackSpline2D track_;
  bool has_track_ = false;

  // This flag controls whether we already have a stored previous solution
  // usable for warm-start rollout.
  bool initialized_ = false;

  // Stored primal trajectory used for warm-start / debug / sampled path output.
  std::vector<std::array<double, NX>> x_stored_;
  std::vector<std::array<double, NU>> u_stored_;

  // External rollout of spline progress used to evaluate kappa(s), v_ref(s)
  // and sampled centerline points outside the solver.
  std::vector<double> s_rollout_;

  int n_rti_iterations_    = 1;
  int n_reset_threshold_   = 3;
  int n_consecutive_fails_ = 0;

  std::vector<double> last_cx_;
  std::vector<double> last_cy_;
};

} // namespace v2_control
