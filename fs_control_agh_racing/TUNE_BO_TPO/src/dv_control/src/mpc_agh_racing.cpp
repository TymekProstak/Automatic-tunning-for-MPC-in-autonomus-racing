// ===============================
// mpc_interface.cpp (SPLIT-V, VARIANT-B + kappa from TrackSpline2D via s-rollout)
//
// Core idea:
// - solve() dostaje s0 (projekcja/estimate z zewnątrz)
// - wewnątrz ltv_matrixes... propaguję s[k] RK4 z s_dot (Frenet)
// - kappa_k = track.getCurvature(s[k])  (closed track assumed)
// - epsi_dot = r - kappa_k * s_dot
// - s_dot    = (v_vehicle*cos(epsi) - vy*sin(epsi)) / (1 - kappa_k*ey)
//
// Notes:
// - velocity_ref służy TYLKO do fallback a_long (finite diff) i do vx0 fallback,
//   NIE jest używane w epsi_dot jako v_path.
// - V_SAFE tylko do atan2 slip.
// - FREN_DENOM_MIN chroni 1 - kappa*ey przed osobliwością (Frenet chart).
// ===============================

#include "mpc_interface_agh_racing.hpp"
#include "spline.hpp"   // <-- dostosuj nazwę jeśli masz inaczej

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <unsupported/Eigen/MatrixFunctions>

#include <sstream>
#include <iomanip>
#include <limits>

extern "C" {
    #include "acados_solver_mpc_ltv_discrete.h"
}

namespace v2_control {

// ============================================================
// Safety for slip computations only
// ============================================================
static constexpr double V_SAFE = 1.0;

static inline double v_safe_for_slip(double v_vehicle)
{
    const double vmag = std::abs(v_vehicle);
    return (vmag < V_SAFE) ? V_SAFE : vmag;
}

// ============================================================
// Safety for Frenet denom: 1 - kappa*ey
// ============================================================
static constexpr double FREN_DENOM_MIN = 0.2;

static inline double frenet_denom_safe(double denom_raw)
{
    return (denom_raw < FREN_DENOM_MIN) ? FREN_DENOM_MIN : denom_raw;
}

// If denom is clamped, derivatives w.r.t ey via denom should be 0 (consistency).
static inline bool frenet_denom_is_clamped(double denom_raw)
{
    return (denom_raw < FREN_DENOM_MIN);
}

// ============================================================
// Track s wrapping/clamping helper
// ============================================================
static inline double normalize_s_on_track(const TrackSpline2D& track, double s)
{
    const double L = track.totalLength();
    if (L <= 1e-12) return 0.0;

    if (track.isClosed()) {
        s = std::fmod(s, L);
        if (s < 0.0) s += L;
        return s;
    } else {
        if (s < 0.0) return 0.0;
        if (s > L)   return L;
        return s;
    }
}

// ============================================================
// Frenet s_dot (variant B) helper
// s_dot = (v_vehicle*cos(epsi) - vy*sin(epsi)) / (1 - kappa*ey)
// ============================================================
static inline double frenet_s_dot(
    double ey,
    double epsi,
    double vy,
    double v_vehicle,
    double kappa)
{
    const double denom_raw = 1.0 - kappa * ey;
    const double denom     = frenet_denom_safe(denom_raw);

    const double s_epsi = std::sin(epsi);
    const double c_epsi = std::cos(epsi);

    const double v_par = v_vehicle * c_epsi - vy * s_epsi;
    return v_par / denom;
}

// ============================================================
// DIAG helpers
// ============================================================
static inline bool eigen_all_finite(const Eigen::MatrixXd& M) {
    return (M.array().isFinite()).all();
}
static inline bool eigen_all_finite_vec(const Eigen::VectorXd& v) {
    return (v.array().isFinite()).all();
}

static inline void dump_acados_solver_stats(ocp_nlp_solver* nlp_solver, int status)
{
    int sqp_iter = -1;
    double time_tot   = std::numeric_limits<double>::quiet_NaN();
    double cost_value = std::numeric_limits<double>::quiet_NaN();

    ocp_nlp_get(nlp_solver, "sqp_iter",   &sqp_iter);
    ocp_nlp_get(nlp_solver, "time_tot",   &time_tot);
    ocp_nlp_get(nlp_solver, "cost_value", &cost_value);

    ROS_WARN_STREAM(std::fixed << std::setprecision(6)
        << "[MPC DIAG] status=" << status
        << " | sqp_iter=" << sqp_iter
        << " | time_tot=" << time_tot
        << " | cost=" << cost_value
    );
}

static inline void dump_solution_preview(ocp_nlp_config* nlp_config,
                                         ocp_nlp_dims* nlp_dims,
                                         ocp_nlp_out* nlp_out,
                                         int N_hor,
                                         int max_k)
{
    max_k = std::min(max_k, N_hor);
    double xk[NX];
    double uk[NU];

    for (int k = 0; k <= max_k; ++k) {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "x", xk);
        if (k < N_hor) ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "u", uk);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "[MPC DIAG] k=" << k << " x=[";
        for (int i = 0; i < NX; ++i) {
            oss << xk[i];
            if (i + 1 < NX) oss << ", ";
        }
        oss << "]";
        if (k < N_hor) oss << " u=[" << uk[0] << ", " << uk[1] << "]";
        ROS_WARN_STREAM(oss.str());
    }
}

static inline void dump_solution_maxima(ocp_nlp_config* nlp_config,
                                        ocp_nlp_dims* nlp_dims,
                                        ocp_nlp_out* nlp_out,
                                        int N_hor)
{
    double xk[NX];
    double uk[NU];

    double max_abs_u_ddreq = 0.0;
    double max_abs_u_mz_tv = 0.0;

    double max_abs_ey = 0.0;
    double max_abs_epsi = 0.0;
    double max_abs_vy = 0.0;
    double max_abs_r = 0.0;
    double max_abs_delta = 0.0;
    double max_abs_d_delta = 0.0;
    double max_abs_d_req = 0.0;

    for (int k = 0; k <= N_hor; ++k)
    {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "x", xk);

        max_abs_ey      = std::max(max_abs_ey,      std::abs(xk[0]));
        max_abs_epsi    = std::max(max_abs_epsi,    std::abs(xk[1]));
        max_abs_vy      = std::max(max_abs_vy,      std::abs(xk[2]));
        max_abs_r       = std::max(max_abs_r,       std::abs(xk[3]));
        max_abs_delta   = std::max(max_abs_delta,   std::abs(xk[4]));
        max_abs_d_delta = std::max(max_abs_d_delta, std::abs(xk[5]));
        max_abs_d_req   = std::max(max_abs_d_req,   std::abs(xk[6]));

        if (k < N_hor) {
            ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, k, "u", uk);
            max_abs_u_ddreq = std::max(max_abs_u_ddreq, std::abs(uk[0]));
            max_abs_u_mz_tv = std::max(max_abs_u_mz_tv, std::abs(uk[1]));
        }
    }

    ROS_WARN_STREAM(std::fixed << std::setprecision(6)
        << "[MPC DIAG] maxima over horizon:"
        << " max|u_ddreq|=" << max_abs_u_ddreq
        << " max|u_mz_tv|=" << max_abs_u_mz_tv
        << " max|ey|=" << max_abs_ey
        << " max|epsi|=" << max_abs_epsi
        << " max|vy|=" << max_abs_vy
        << " max|r|=" << max_abs_r
        << " max|delta|=" << max_abs_delta
        << " max|d_delta|=" << max_abs_d_delta
        << " max|d_req|=" << max_abs_d_req
    );
}

// ============================================================
// exact discretization via expm: [A B; 0 0]
// ============================================================
static inline void discretize_expm_AB(
    const Eigen::Matrix<double, NX, NX>& A,
    const Eigen::Matrix<double, NX, NU>& B,
    double dt,
    Eigen::Matrix<double, NX, NX>& Ad,
    Eigen::Matrix<double, NX, NU>& Bd)
{
    Eigen::Matrix<double, NX + NU, NX + NU> M = Eigen::Matrix<double, NX + NU, NX + NU>::Zero();
    M.block<NX, NX>(0, 0)  = A;
    M.block<NX, NU>(0, NX) = B;

    Eigen::Matrix<double, NX + NU, NX + NU> E = (M * dt).exp();
    Ad = E.block<NX, NX>(0, 0);
    Bd = E.block<NX, NU>(0, NX);
}

// ============================================================
// Continuous dynamics — SPLIT-V, VARIANT-B
// kappa -> from track at s[k]
// v_vehicle -> from longitudinal prediction
// ============================================================
static inline Eigen::Matrix<double, NX, 1> f_splitv_continuous(
    const Eigen::Matrix<double, NX, 1>& x,
    const Eigen::Matrix<double, NU, 1>& u,
    double kappa,
    double v_vehicle,
    const ParamBank& param)
{
    const double m  = param.get("model_m");
    const double Iz = param.get("model_Iz");
    const double lf = param.get("model_lf");
    const double lr = param.get("model_lr");
    const double B  = param.get("model_B");
    const double C  = param.get("model_C");
    const double D  = param.get("model_D");
    const double Cl = param.get("model_Cl");
    const double g  = 9.81;

    const double omega    = param.get("model_steer_natural_freq");
    const double damp     = param.get("model_steer_damping");
    const double omega_sq = omega * omega;

    const double v_slip = v_safe_for_slip(v_vehicle);

    const double ey      = x(0);
    const double epsi    = x(1);
    const double vy      = x(2);
    const double r       = x(3);
    const double delta   = x(4);
    const double d_delta = x(5);
    const double d_req   = x(6);

    const double u_ddreq = u(0);
    const double u_mz_tv = u(1);

    const double s_epsi  = std::sin(epsi);
    const double c_epsi  = std::cos(epsi);
    const double c_delta = std::cos(delta);

    // Frenet s_dot
    const double s_dot = frenet_s_dot(ey, epsi, vy, v_vehicle, kappa);

    const double yf = vy + lf * r;
    const double yr = vy - lr * r;

    const double alpha_f = delta - std::atan2(yf, v_slip);
    const double alpha_r = -std::atan2(yr, v_slip);

    const double F_N_net = m * g + Cl * v_vehicle * v_vehicle;
    const double F_N_F = F_N_net * lr / (lf + lr);
    const double F_N_R = F_N_net * lf / (lf + lr);

    const double Fyf = F_N_F * D * std::sin(C * std::atan(B * alpha_f));
    const double Fyr = F_N_R * D * std::sin(C * std::atan(B * alpha_r));

    Eigen::Matrix<double, NX, 1> xdot;
    xdot.setZero();

    // ey_dot
    xdot(0) = vy * c_epsi + v_vehicle * s_epsi;

    // epsi_dot (variant B)
    xdot(1) = r - kappa * s_dot;

    // vy_dot
    xdot(2) = (Fyf * c_delta + Fyr) / m - v_vehicle * r;

    // r_dot + TV yaw moment
    xdot(3) = (lf * Fyf * c_delta - lr * Fyr) / Iz + (u_mz_tv / Iz);

    // steering actuator chain
    xdot(4) = d_delta;
    xdot(5) = -omega_sq * delta - 2.0 * damp * omega * d_delta + omega_sq * d_req;

    // d_req_dot
    xdot(6) = u_ddreq;

    return xdot;
}

// ============================================================
// RK4 step for x AND s (kappa frozen over dt)
// ============================================================
static inline void rk4_step_splitv_xs(
    const Eigen::Matrix<double, NX, 1>& x,
    const Eigen::Matrix<double, NU, 1>& u,
    double s,
    double kappa,
    double v_vehicle,
    double dt,
    const TrackSpline2D& track,
    const ParamBank& param,
    Eigen::Matrix<double, NX, 1>& x_next,
    double& s_next)
{
    const auto k1 = f_splitv_continuous(x,                 u, kappa, v_vehicle, param);
    const auto x2 = x + 0.5*dt*k1;
    const auto k2 = f_splitv_continuous(x2,                u, kappa, v_vehicle, param);
    const auto x3 = x + 0.5*dt*k2;
    const auto k3 = f_splitv_continuous(x3,                u, kappa, v_vehicle, param);
    const auto x4 = x + dt*k3;
    const auto k4 = f_splitv_continuous(x4,                u, kappa, v_vehicle, param);

    x_next = x + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);

    // s RK4 using s_dot(x_stage), kappa frozen
    const double sdot1 = frenet_s_dot(x(0),  x(1),  x(2),  v_vehicle, kappa);
    const double sdot2 = frenet_s_dot(x2(0), x2(1), x2(2), v_vehicle, kappa);
    const double sdot3 = frenet_s_dot(x3(0), x3(1), x3(2), v_vehicle, kappa);
    const double sdot4 = frenet_s_dot(x4(0), x4(1), x4(2), v_vehicle, kappa);

    s_next = s + (dt/6.0) * (sdot1 + 2.0*sdot2 + 2.0*sdot3 + sdot4);
    s_next = normalize_s_on_track(track, s_next);
}

// ============================================================
// Helper: a_long fallback from v_ref (finite difference)
// ============================================================
static inline std::vector<double> build_a_long_from_v_ref(
    int N_hor,
    double dt,
    const std::vector<double>& v_ref)
{
    std::vector<double> a;
    a.assign(N_hor, 0.0);

    if ((int)v_ref.size() < 2 || dt <= 0.0) return a;

    for (int k = 0; k < N_hor; ++k) {
        const int k0 = std::min(k, (int)v_ref.size() - 2);
        const int k1 = k0 + 1;
        const double dv = v_ref[k1] - v_ref[k0];
        double ak = dv / dt;
        if (!std::isfinite(ak)) ak = 0.0;
        a[k] = ak;
    }
    return a;
}

// ============================================================
// Helper: build predicted v_vehicle[k] from vx0 + a_long_ref
// ============================================================
static inline std::vector<double> build_v_vehicle_pred(
    int N_hor,
    double dt,
    double vx0_body,
    const std::vector<double>& a_long_ref)
{
    std::vector<double> vveh;
    vveh.resize(N_hor);

    double v = std::max(0.0, vx0_body);

    for (int k = 0; k < N_hor; ++k) {
        vveh[k] = std::max(0.0, v);

        if (k < (int)a_long_ref.size()) {
            double a = a_long_ref[k];
            if (!std::isfinite(a)) a = 0.0;
            v += a * dt;
        }
    }
    return vveh;
}

// ============================================================
// Constructors / Destructor
// ============================================================
MPCInterface::MPCInterface()
{
    std::cout << "[MPCInterface] DEFAULT ctor – solver NOT initialized properly!" << std::endl;
    capsule_    = nullptr;
    nlp_config_ = nullptr;
    nlp_dims_   = nullptr;
    nlp_in_     = nullptr;
    nlp_out_    = nullptr;
    nlp_solver_ = nullptr;
    is_initialized_ = false;
    last_output.assign(N, Eigen::Matrix<double, NU, 1>::Zero());
}

MPCInterface::MPCInterface(const ParamBank &P)
{
    ROS_INFO("[MPCInterface] Constructing MPCInterface with ParamBank");

    capsule_ = mpc_ltv_discrete_acados_create_capsule();
    if (!capsule_) {
        ROS_ERROR("[MPC SOLVER] create_capsule returned NULL!");
        return;
    }

    const int status = mpc_ltv_discrete_acados_create(capsule_);
    if (status != 0) {
        ROS_ERROR_STREAM("[MPC SOLVER] Could not create ACADOS solver! Status: " << status);
    } else {
        nlp_config_ = mpc_ltv_discrete_acados_get_nlp_config(capsule_);
        nlp_dims_   = mpc_ltv_discrete_acados_get_nlp_dims(capsule_);
        nlp_in_     = mpc_ltv_discrete_acados_get_nlp_in(capsule_);
        nlp_out_    = mpc_ltv_discrete_acados_get_nlp_out(capsule_);
        nlp_solver_ = mpc_ltv_discrete_acados_get_nlp_solver(capsule_);
        ROS_INFO_STREAM("[MPC SOLVER] Solver created successfully.");
    }

    param_ = P;
    is_initialized_ = false;
    last_output.assign(N, Eigen::Matrix<double, NU, 1>::Zero());
}

MPCInterface::~MPCInterface()
{
    ROS_INFO("[MPCInterface] Destructor called");
    if (capsule_) {
        ROS_INFO("[MPCInterface] Freeing ACADOS solver and capsule");
        mpc_ltv_discrete_acados_free(capsule_);
        mpc_ltv_discrete_acados_free_capsule(capsule_);
        capsule_ = nullptr;
    }
}

// ============================================================
// reset_initial_guess — SPLIT-V (x only)
// ============================================================
void MPCInterface::reset_initial_guess_splitv_(const MPC_State& x0,
                                              const std::vector<double>& v_vehicle_pred)
{
    ROS_INFO("[MPCInterface] Resetting initial guess (Straight Line Prediction, split-v)");
    if (!nlp_config_ || !nlp_dims_ || !nlp_in_ || !nlp_out_) return;

    const double dt = 1.0 / param_.get("odom_frequency");
    const double v_fallback = param_.get("v_target");

    const double current_epsi = x0.epsi;

    double x_traj[NX];
    double u_zero[NU] = {0.0,0.0};

    double ey_pred = x0.ey;

    for (int i = 0; i <= N; ++i) {
        double v_i = v_fallback;
        if (!v_vehicle_pred.empty()) {
            const int idx = std::min(i, (int)v_vehicle_pred.size() - 1);
            if (std::isfinite(v_vehicle_pred[idx])) v_i = v_vehicle_pred[idx];
        }

        if (i > 0) {
            ey_pred += (v_i * std::sin(current_epsi)) * dt;
        }

        x_traj[0] = ey_pred;
        x_traj[1] = current_epsi;
        x_traj[2] = 0.0;
        x_traj[3] = 0.0;
        x_traj[4] = 0.0;
        x_traj[5] = 0.0;
        x_traj[6] = 0.0;

        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", x_traj);
        if (i < N) ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_zero);
    }
}

// ============================================================
// Jacobian — SPLIT-V, VARIANT-B
// signature: 2nd argument = kappa (NOT v_path)
// ============================================================
void MPCInterface::calculate_continuous_jacobian_splitv_(
    const Eigen::Matrix<double, NX, 1>& x,
    double kappa,
    double v_vehicle,
    Eigen::Matrix<double, NX, NX>& Ac,
    Eigen::Matrix<double, NX, NU>& Bc)
{
    const double m  = param_.get("model_m");
    const double Iz = param_.get("model_Iz");
    const double lf = param_.get("model_lf");
    const double lr = param_.get("model_lr");
    const double B  = param_.get("model_B");
    const double C  = param_.get("model_C");
    const double D  = param_.get("model_D");
    const double Cl = param_.get("model_Cl");
    const double g  = 9.81;

    const double omega    = param_.get("model_steer_natural_freq");
    const double damp     = param_.get("model_steer_damping");
    const double omega_sq = omega * omega;

    const double v_slip = v_safe_for_slip(v_vehicle);

    const double ey    = x(0);
    const double epsi  = x(1);
    const double vy    = x(2);
    const double r     = x(3);
    const double delta = x(4);

    const double s_epsi  = std::sin(epsi);
    const double c_epsi  = std::cos(epsi);
    const double s_delta = std::sin(delta);
    const double c_delta = std::cos(delta);

    const double yf = vy + lf * r;
    const double yr = vy - lr * r;

    const double alpha_f = delta - std::atan2(yf, v_slip);
    const double alpha_r = -std::atan2(yr, v_slip);

    const double F_N_net = m * g + Cl * v_vehicle * v_vehicle;
    const double F_N_F = F_N_net * lr / (lf + lr);
    const double F_N_R = F_N_net * lf / (lf + lr);

    const double Fyf = F_N_F * D * std::sin(C * std::atan(B * alpha_f));
    const double Fyr = F_N_R * D * std::sin(C * std::atan(B * alpha_r));

    const double denom_f = v_slip*v_slip + yf*yf;
    const double denom_r = v_slip*v_slip + yr*yr;

    const double datan_f_dyf = v_slip / denom_f;
    const double datan_r_dyr = v_slip / denom_r;

    const double d_alpha_f_d_vy    = -datan_f_dyf;
    const double d_alpha_f_d_r     = -datan_f_dyf * lf;
    const double d_alpha_f_d_delta =  1.0;

    const double d_alpha_r_d_vy = -datan_r_dyr;
    const double d_alpha_r_d_r  =  datan_r_dyr * lr;

    const double dFyf_d_alpha_f = F_N_F * D * std::cos(C * std::atan(B * alpha_f)) * C * B / (1.0 + B * B * alpha_f * alpha_f);
    const double dFyr_d_alpha_r = F_N_R * D * std::cos(C * std::atan(B * alpha_r)) * C * B / (1.0 + B * B * alpha_r * alpha_r);

    const double dFyf_d_vy    = dFyf_d_alpha_f * d_alpha_f_d_vy;
    const double dFyf_d_r     = dFyf_d_alpha_f * d_alpha_f_d_r;
    const double dFyf_d_delta = dFyf_d_alpha_f * d_alpha_f_d_delta;

    const double dFyr_d_vy = dFyr_d_alpha_r * d_alpha_r_d_vy;
    const double dFyr_d_r  = dFyr_d_alpha_r * d_alpha_r_d_r;

    Ac.setZero();
    Bc.setZero();

    // ey_dot = vy*cos(epsi) + v_vehicle*sin(epsi)
    Ac(0, 1) = -vy * s_epsi + v_vehicle * c_epsi;
    Ac(0, 2) =  c_epsi;

    // epsi_dot = r - kappa*s_dot
    // s_dot = (v_vehicle*cos(epsi) - vy*sin(epsi)) / (1 - kappa*ey)
    const double denom_raw = 1.0 - kappa * ey;
    const bool denom_clamped = frenet_denom_is_clamped(denom_raw);
    const double denom = frenet_denom_safe(denom_raw);

    const double n = v_vehicle * c_epsi - vy * s_epsi; // numerator

    // d(epsi_dot)/d(ey): only if denom not clamped (consistency with clamp)
    if (!denom_clamped) {
        Ac(1, 0) = -(kappa * kappa) * n / (denom * denom);
    } else {
        Ac(1, 0) = 0.0;
    }

    // d(epsi_dot)/d(epsi)
    Ac(1, 1) = (kappa / denom) * (v_vehicle * s_epsi + vy * c_epsi);

    // d(epsi_dot)/d(vy)
    Ac(1, 2) = (kappa / denom) * (s_epsi);

    // d(epsi_dot)/d(r)
    Ac(1, 3) = 1.0;

    // vy_dot = (Fyf*cos(delta)+Fyr)/m - v_vehicle*r
    Ac(2, 2) = (1.0 / m) * (dFyf_d_vy * c_delta + dFyr_d_vy);
    Ac(2, 3) = (1.0 / m) * (dFyf_d_r  * c_delta + dFyr_d_r) - v_vehicle;

    const double d_vy_term_d_delta = dFyf_d_delta * c_delta - Fyf * s_delta;
    Ac(2, 4) = (1.0 / m) * d_vy_term_d_delta;

    // r_dot = (lf*Fyf*cos(delta) - lr*Fyr)/Iz + Mz_tv/Iz
    Ac(3, 2) = (1.0 / Iz) * (lf * dFyf_d_vy * c_delta - lr * dFyr_d_vy);
    Ac(3, 3) = (1.0 / Iz) * (lf * dFyf_d_r  * c_delta - lr * dFyr_d_r);

    const double d_r_term_d_delta = lf * (dFyf_d_delta * c_delta - Fyf * s_delta);
    Ac(3, 4) = (1.0 / Iz) * d_r_term_d_delta;

    // actuator states
    Ac(4, 5) = 1.0;
    Ac(5, 4) = -omega_sq;
    Ac(5, 5) = -2.0 * damp * omega;
    Ac(5, 6) =  omega_sq;

    // u = d(d_req)/dt
    Bc(6, 0) = 1.0;
    // u(1) = Mz_tv -> r_dot += Mz_tv / Iz
    Bc(3, 1) = 1.0 / Iz;
}

// ============================================================
// LTV matrices — SPLIT-V (kappa from track via s-rollout)
// ============================================================
void MPCInterface::ltv_matrixes_to_acados_splitv_(
    const MPC_State& x0,
    const TrackSpline2D& track,
    double s0,
    const std::vector<double>& v_vehicle_vec)
{
    const double dt = 1.0 / param_.get("odom_frequency");
    const double v_vehicle_fallback = param_.get("v_target");

    // normalize s0 to track domain
    double s = normalize_s_on_track(track, s0);

    // x trajectory
    std::vector<Eigen::Matrix<double, NX, 1>> x_traj(N + 1);
    x_traj[0] << x0.ey, x0.epsi, x0.vy, x0.r, x0.delta, x0.d_delta, x0.delta_request;

    // s trajectory (for kappa lookup)
    std::vector<double> s_traj(N + 1, s);
    s_traj[0] = s;

    for (int k = 0; k < N; ++k)
    {
        Eigen::Matrix<double, NU, 1> u_k;
        u_k.setZero();
        if (!last_output.empty()) {
            int idx = std::min(k, (int)last_output.size() - 1);
            u_k = last_output[idx];
        }

        double v_vehicle = v_vehicle_fallback;
        if (!v_vehicle_vec.empty()) {
            const int vidx = std::min(k, (int)v_vehicle_vec.size() - 1);
            if (std::isfinite(v_vehicle_vec[vidx])) v_vehicle = v_vehicle_vec[vidx];
        }
        v_vehicle = std::max(0.0, v_vehicle);

        const double s_k = s_traj[k];
        const double kappa_k = track.getCurvature(s_k);

        const Eigen::Matrix<double, NX, 1> x_curr = x_traj[k];

        Eigen::Matrix<double, NX, NX> Ac_cont;
        Eigen::Matrix<double, NX, NU> Bc_cont;
        calculate_continuous_jacobian_splitv_(x_curr, kappa_k, v_vehicle, Ac_cont, Bc_cont);

        Eigen::Matrix<double, NX, NX> Ad;
        Eigen::Matrix<double, NX, NU> Bd;
        discretize_expm_AB(Ac_cont, Bc_cont, dt, Ad, Bd);

        // rollout x and s with the same frozen kappa_k over dt
        Eigen::Matrix<double, NX, 1> x_next;
        double s_next = s_k;
        rk4_step_splitv_xs(x_curr, u_k, s_k, kappa_k, v_vehicle, dt, track, param_, x_next, s_next);

        x_traj[k + 1] = x_next;
        s_traj[k + 1] = s_next;

        // affine term
        Eigen::Matrix<double, NX, 1> Kd = x_next - (Ad * x_curr + Bd * u_k);

        {
            Eigen::MatrixXd Ad_d = Ad;
            Eigen::MatrixXd Bd_d = Bd;
            Eigen::VectorXd Kd_d = Kd;
            Eigen::VectorXd xn_d = x_next;
            Eigen::VectorXd xc_d = x_curr;

            if (!eigen_all_finite(Ad_d) || !eigen_all_finite(Bd_d) ||
                !eigen_all_finite_vec(Kd_d) || !eigen_all_finite_vec(xn_d) || !eigen_all_finite_vec(xc_d))
            {
                ROS_ERROR_STREAM("[MPC DIAG] NaN/INF in LTV params at stage k=" << k
                    << " | s=" << s_k
                    << " | kappa=" << kappa_k
                    << " | u_k=" << u_k
                    << " | v_vehicle=" << v_vehicle
                    << " | x_curr=" << x_curr.transpose());

                ROS_ERROR_STREAM("[MPC DIAG] norms: ||Ad||=" << Ad.norm()
                    << " ||Bd||=" << Bd.norm()
                    << " ||Kd||=" << Kd.norm()
                    << " ||x_next||=" << x_next.norm());
            }
        }

        std::vector<double> p_vec;
        p_vec.reserve(NX*NX + NX*NU + NX);

        for (int c = 0; c < NX; ++c) for (int r0 = 0; r0 < NX; ++r0) p_vec.push_back(Ad(r0, c));
        for (int c = 0; c < NU; ++c) for (int r0 = 0; r0 < NX; ++r0) p_vec.push_back(Bd(r0, c));
        for (int r0 = 0; r0 < NX; ++r0) p_vec.push_back(Kd(r0));

        mpc_ltv_discrete_acados_update_params(capsule_, k, p_vec.data(), (int)p_vec.size());
    }
}

// ============================================================
// Costs (unchanged)
// ============================================================
void MPCInterface::set_cost_to_acados()
{
    if (!nlp_config_ || !nlp_dims_ || !nlp_in_) return;

    const double Q_y      = param_.get("mpc_cost_Q_y");
    const double Q_psi    = param_.get("mpc_cost_Q_psi");
    const double R_ddelta = param_.get("mpc_cost_R_ddelta");
    const double Q_r      = param_.get("mpc_cost_Q_r");
    const double Q_delta  = param_.get("mpc_cost_Q_delta");
    const double R_tv     = param_.get("mpc_cost_R_tv");

    const double term_scale = 1.0;

    const int ny = NX + NU;
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(ny, ny);
    W(0, 0)       = Q_y;
    W(1, 1)       = Q_psi;
    W(3, 3)       = Q_r;
    W(4, 4)       = Q_delta;
    W(NX, NX)     = R_ddelta;
    W(NX+1, NX+1) = R_tv;

    const int ny_e = NX;
    Eigen::MatrixXd W_e = Eigen::MatrixXd::Zero(ny_e, ny_e);
    W_e(0, 0) = Q_y   * term_scale;
    W_e(1, 1) = Q_psi * term_scale;

    for (int i = 0; i < N; ++i) ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data());
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N, "W", W_e.data());
}

// ============================================================
// SOLVE — SPLIT-V (VARIANT-B + kappa from track + s0)
// ============================================================
MPC_Return MPCInterface::solve(const MPC_State &x0,
                               const TrackSpline2D& track,
                               double s0,
                               const Eigen::VectorXd &velocity_ref,
                               const Eigen::VectorXd &acceleration_ref,
                               double vx_body)
{
    if (!capsule_ || !nlp_config_ || !nlp_dims_ || !nlp_in_ || !nlp_out_ || !nlp_solver_) {
        ROS_ERROR_THROTTLE(1.0, "[MPCInterface::solve] Solver structures NOT initialized!");
        return {0.0, 0.0, false, x0.r};
    }

    if (!track.valid()) {
        ROS_ERROR_THROTTLE(1.0, "[MPCInterface::solve] TrackSpline2D invalid!");
        return {0.0, 0.0, false, x0.r};
    }

    const double dt = 1.0 / param_.get("odom_frequency");

    // v_ref (for longitudinal pred only)
    std::vector<double> v_ref_vec(velocity_ref.data(),
                                  velocity_ref.data() + velocity_ref.size());

    std::vector<double> a_long_vec(acceleration_ref.data(),
                                   acceleration_ref.data() + acceleration_ref.size());

    // ============================================================
    // Resize v_ref to N (only for fallback)
    // ============================================================
    if ((int)v_ref_vec.size() != N) {
        ROS_WARN_STREAM("[MPCInterface::solve] Velocity(v_ref) size mismatch! Expected "
                        << N << ", got " << v_ref_vec.size() << " -> resizing.");
        const double v_fallback = param_.get("v_target");
        if ((int)v_ref_vec.size() < N) v_ref_vec.resize(N, v_ref_vec.empty() ? v_fallback : v_ref_vec.back());
        if ((int)v_ref_vec.size() > N) v_ref_vec.resize(N);
    }

    // if a_long empty -> fallback from v_ref finite diff
    if (a_long_vec.empty()) {
        a_long_vec = build_a_long_from_v_ref(N, dt, v_ref_vec);
    } else if ((int)a_long_vec.size() != N) {
        if ((int)a_long_vec.size() < N) a_long_vec.resize(N, a_long_vec.empty() ? 0.0 : a_long_vec.back());
        if ((int)a_long_vec.size() > N) a_long_vec.resize(N);
    }

    // vx0_body: prefer odom, fallback to v_ref[0]
    double vx0_body = vx_body;
    if (!std::isfinite(vx0_body) || vx0_body <= 0.0) {
        vx0_body = (v_ref_vec.empty() ? param_.get("v_target") : v_ref_vec.front());
    }

    // predicted v_vehicle[k]
    std::vector<double> v_vehicle_vec = build_v_vehicle_pred(N, dt, vx0_body, a_long_vec);

    // ============================================================
    // Warm-start init
    // ============================================================
    if (!is_initialized_) {
        mpc_ltv_discrete_acados_reset(capsule_, 1);
        reset_initial_guess_splitv_(x0, v_vehicle_vec);
        last_output.assign(N, Eigen::Matrix<double, NU, 1>::Zero());
        is_initialized_ = true;
    }

    // ============================================================
    // Fix x0
    // ============================================================
    double x0_arr[NX];
    x0.to_array(x0_arr);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", x0_arr);
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", x0_arr);

    // ============================================================
    // LTV params + cost
    // ============================================================
    ltv_matrixes_to_acados_splitv_(x0, track, s0, v_vehicle_vec);
    set_cost_to_acados();

    // ============================================================
    // Solve
    // ============================================================
    const int status = mpc_ltv_discrete_acados_solve(capsule_);

    if (status == 0)
    {
        std::vector<Eigen::Matrix<double, NU, 1>> new_u_traj;
        new_u_traj.reserve(N);

        double u_step_arr[NU];

        for (int i = 0; i < N; ++i) {
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "u", u_step_arr);
            Eigen::Matrix<double, NU, 1> u_step;
            u_step << u_step_arr[0], u_step_arr[1];
            new_u_traj.push_back(u_step);
        }

        // Shift warm-start for u
        last_output.clear();
        last_output.reserve(N);
        for (int i = 1; i < N; ++i) last_output.push_back(new_u_traj[i]);
        last_output.push_back(new_u_traj.back());

        const double u_ddreq_final = new_u_traj[0](0);
        const double u_mz_tv_final = new_u_traj[0](1);

        if (!std::isfinite(u_ddreq_final) || !std::isfinite(u_mz_tv_final)) {
            ROS_ERROR("[MPC DIAG] Solver returned NaN/INF in first control!");
            dump_acados_solver_stats(nlp_solver_, status);
            dump_solution_maxima(nlp_config_, nlp_dims_, nlp_out_, N);
            dump_solution_preview(nlp_config_, nlp_dims_, nlp_out_, N, std::min(10, N));

            is_initialized_ = false;
            last_output.assign(N, Eigen::Matrix<double, NU, 1>::Zero());
            return {0.0, 0.0, false, x0_arr[3]};
        }

        // Next predicted yaw-rate from x at k=1
        double x1[NX];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x1);
        double next_r = x1[3];
        if (!std::isfinite(next_r)) {
            ROS_ERROR("[MPC DIAG] next_predicted_yaw_rate is NaN/INF!");
            next_r = 0.0;
        }

        return {u_ddreq_final, u_mz_tv_final, true, next_r};
    }

    // Fail path
    ROS_WARN_STREAM("[MPC] Solver FAILED with status: " << status);

    dump_acados_solver_stats(nlp_solver_, status);
    dump_solution_maxima(nlp_config_, nlp_dims_, nlp_out_, N);
    dump_solution_preview(nlp_config_, nlp_dims_, nlp_out_, N, std::min(10, N));

    is_initialized_ = false;
    last_output.assign(N, Eigen::Matrix<double, NU, 1>::Zero());
    return {0.0, 0.0, false, x0_arr[3]};
}

// ============================================================
// Convenience overload: no acceleration_ref -> fallback from v_ref
// ============================================================
MPC_Return MPCInterface::solve(const MPC_State &x0,
                               const TrackSpline2D& track,
                               double s0,
                               const Eigen::VectorXd &velocity_ref)
{
    Eigen::VectorXd a_empty; // empty -> fallback in main solve()
    return solve(x0, track, s0, velocity_ref, a_empty, velocity_ref(0));
}

// ============================================================
// Stubs
// ============================================================
void MPCInterface::print_problem_debug(const Eigen::Matrix<double, NX, NX> &,
                                       const Eigen::Matrix<double, NX, NU> &,
                                       const Eigen::Matrix<double, NX, 1>  &,
                                       const MPC_State &)
{
}

void MPCInterface::build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>&,
                                                 Eigen::Matrix<double, NX, NU>&) const
{
}

void MPCInterface::build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>&,
                                                 Eigen::Matrix<double, NX, NU>&,
                                                 double) const
{
}

void MPCInterface::push_lti_params_to_acados(double)
{
}

} // namespace v2_control