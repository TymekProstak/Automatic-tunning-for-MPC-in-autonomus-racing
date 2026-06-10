#include "mpc_interface_ka_racing.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace v2_control {

// ============================================================
// Helper: exact discretization via matrix exponential (A,B)->(Ad,Bd)
// (full augmented expm: [A B; 0 0])
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

// =========================
// Timing helpers (local)
// =========================
static inline double ms_since(const std::chrono::steady_clock::time_point& t0,
                              const std::chrono::steady_clock::time_point& t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct SectionStats {
    double sum_ms = 0.0;
    double max_ms = 0.0;
    void add(double x) {
        sum_ms += x;
        if (x > max_ms) max_ms = x;
    }
    void reset() { sum_ms = 0.0; max_ms = 0.0; }
};

// =====================================
// KONSTRUKTOR
// =====================================
MPCInterface::MPCInterface()
: N_(15), is_initialized_(false)
{
    last_output.assign(N_, Eigen::Matrix<double, NU, 1>::Zero());
}

MPCInterface::MPCInterface(const ParamBank& P)
: param_(P), N_(15), is_initialized_(false)
{
    try {
        N_ = static_cast<int>(param_.get("mpc_N"));
    } catch (...) {
        N_ = 15;
    }

    N_ = std::max(1, N_);
    last_output.assign(N_, Eigen::Matrix<double, NU, 1>::Zero());
}

MPCInterface::~MPCInterface() = default;

// =====================================
// RESET INITIAL GUESS (no-op)
// =====================================
void MPCInterface::reset_initial_guess(const MPC_State& /*x0*/,
                                       const std::vector<double>& /*vref_vec*/)
{
    // Bez ACADOS nic nie “wpycham” — solver jest closed-form.
}

// ============================================================
// LTI continuous model around straight driving (small angles, linear tires)
// u[0] = d(delta_req)/dt
// Stan: [ey, epsi, vy, r, delta, delta_dot, delta_req]
// ============================================================
void MPCInterface::build_lti_continuous_matrices(
    Eigen::Matrix<double, NX, NX>& Ac,
    Eigen::Matrix<double, NX, NU>& Bc,
    double v_ref0) const
{
    const double m  = param_.get("model_m");
    const double Iz = param_.get("model_Iz");
    const double lf = param_.get("model_lf");
    const double lr = param_.get("model_lr");
    const double Cr = param_.get("model_Cr");
    const double Cf = param_.get("model_Cf");

    const double v_eps = 0.5;
    const double v = std::max(v_ref0, v_eps);

    const double omega    = param_.get("model_steer_natural_freq");
    const double damp     = param_.get("model_steer_damping");
    const double omega_sq = omega * omega;

    Ac.setZero();
    Bc.setZero();

    // ey_dot   ≈ vy + v*epsi
    // epsi_dot ≈ r
    Ac(0, 1) = v;
    Ac(0, 2) = 1.0;
    Ac(1, 3) = 1.0;

    // vy_dot
    Ac(2, 2) = -(Cf + Cr) / (m * v);
    Ac(2, 3) = -(Cf * lf - Cr * lr) / (m * v) - v;
    Ac(2, 4) =  (Cf) / m;

    // r_dot
    Ac(3, 2) = -(lf * Cf - lr * Cr) / (Iz * v);
    Ac(3, 3) = -(lf * lf * Cf + lr * lr * Cr) / (Iz * v);
    Ac(3, 4) =  (lf * Cf) / Iz;

    // Servo (PT2):
    Ac(4, 5) = 1.0;
    Ac(5, 4) = -omega_sq;
    Ac(5, 5) = -2.0 * damp * omega;
    Ac(5, 6) =  omega_sq;

    // u[0] = d(delta_req)/dt
    Bc(6, 0) = 1.0;
}

void MPCInterface::build_lti_continuous_matrices(
    Eigen::Matrix<double, NX, NX>& Ac,
    Eigen::Matrix<double, NX, NU>& Bc) const
{
    const double v_ref0 = param_.get("v_target");
    build_lti_continuous_matrices(Ac, Bc, v_ref0);
}

// =====================================
// SOLVE — unconstrained condensed QP:
//   min 0.5 z^T H z + g^T z   =>   H z = -g
// =====================================
MPC_Return MPCInterface::solve(const MPC_State& x0,
                               const Eigen::VectorXd& /*curvature_ref*/,
                               const Eigen::VectorXd& velocity_ref)
{
    // -------- timing stats (windowed) --------
    static int window_cnt = 0;
    static const int WINDOW_N = 50;

    static SectionStats st_total;
    static SectionStats st_build_Ac;
    static SectionStats st_expm;
    static SectionStats st_weights;
    static SectionStats st_build_Hg;
    static SectionStats st_llt;
    static SectionStats st_solve;
    static SectionStats st_predict;

    const auto t_total0 = std::chrono::steady_clock::now();

    // 0) v_ref0
    double v_ref0 = param_.get("v_target");
    if (velocity_ref.size() > 0 && std::isfinite(velocity_ref(0))) {
        v_ref0 = velocity_ref(0);
    }

    // 1) Discretize
    const double dt = 1.0 / param_.get("odom_frequency");

    const auto t_Ac0 = std::chrono::steady_clock::now();
    Eigen::Matrix<double, NX, NX> Ac;
    Eigen::Matrix<double, NX, NU> Bc;
    build_lti_continuous_matrices(Ac, Bc, v_ref0);
    const auto t_Ac1 = std::chrono::steady_clock::now();

    const auto t_expm0 = std::chrono::steady_clock::now();
    Eigen::Matrix<double, NX, NX> Ad;
    Eigen::Matrix<double, NX, NU> Bd;
    discretize_expm_AB(Ac, Bc, dt, Ad, Bd);
    const auto t_expm1 = std::chrono::steady_clock::now();

    // 2) Weights
    const auto t_w0 = std::chrono::steady_clock::now();

    const double Q_y      = param_.get("mpc_cost_Q_y");
    const double Q_psi    = param_.get("mpc_cost_Q_psi");
    const double Q_r      = param_.get("mpc_cost_Q_r");
    const double R_ddelta = param_.get("mpc_cost_R_ddelta");

    const double term_scale = 1.0;

    Eigen::Matrix<double, NX, NX> Q = Eigen::Matrix<double, NX, NX>::Zero();
    Q(0,0) = Q_y;
    Q(1,1) = Q_psi;

    Eigen::Matrix<double, NX, NX> P = Eigen::Matrix<double, NX, NX>::Zero();
    P(0,0) = Q_y   * term_scale;
    P(1,1) = Q_psi * term_scale;

    const auto t_w1 = std::chrono::steady_clock::now();

    // 3) x0 -> Eigen vec
    double x0_arr[NX];
    x0.to_array(x0_arr);
    Eigen::Matrix<double, NX, 1> x_base;
    for (int i = 0; i < NX; ++i) x_base(i) = x0_arr[i];

    // 4) Build condensed H,g
    const auto t_H0 = std::chrono::steady_clock::now();

    const int nz = N_ * NU;     // NU=1 => nz=N_
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nz, nz);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(nz);

    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(NX, nz);

    for (int k = 0; k < N_; ++k) {
        H(k, k) += R_ddelta;
    }

    for (int k = 0; k < N_; ++k)
    {
        x_base = Ad * x_base;

        S = Ad * S;
        S.block(0, k * NU, NX, NU) += Bd;

        const Eigen::Matrix<double, NX, NX>& Qk = (k == N_ - 1) ? P : Q;

        H.noalias() += S.transpose() * Qk * S;
        g.noalias() += S.transpose() * Qk * x_base;
    }

    const auto t_H1 = std::chrono::steady_clock::now();

    // 5) Solve H z = -g via Cholesky (LLT)
    const double reg = 1e-9;
    H.diagonal().array() += reg;

    const auto t_llt0 = std::chrono::steady_clock::now();
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    const auto t_llt1 = std::chrono::steady_clock::now();

    if (llt.info() != Eigen::Success) {
        ROS_WARN("[MPC] LLT failed -> reset");
        is_initialized_ = false;
        last_output.assign(N_, Eigen::Matrix<double, NU, 1>::Zero());
        return {0.0, 0.0, false, x0.r};
    }

    const auto t_sol0 = std::chrono::steady_clock::now();
    Eigen::VectorXd z = llt.solve(-g);
    const auto t_sol1 = std::chrono::steady_clock::now();

    if (llt.info() != Eigen::Success || !z.allFinite()) {
        ROS_WARN("[MPC] solve produced NaN/Inf -> reset");
        is_initialized_ = false;
        last_output.assign(N_, Eigen::Matrix<double, NU, 1>::Zero());
        return {0.0, 0.0, false, x0.r};
    }

    const double ddelta_opt = z(0);
    const double mtv_opt    = 0.0;

    // next predicted yaw rate
    const auto t_pr0 = std::chrono::steady_clock::now();
    Eigen::Matrix<double, NX, 1> x0v;
    for (int i = 0; i < NX; ++i) x0v(i) = x0_arr[i];

    Eigen::Matrix<double, NX, 1> x1 = Ad * x0v + Bd * ddelta_opt;
    double next_r = x1(3);
    if (!std::isfinite(next_r)) next_r = x0.r;
    const auto t_pr1 = std::chrono::steady_clock::now();

    // save trajectory (optional)
    last_output.clear();
    last_output.reserve(N_);
    for (int k = 0; k < N_; ++k) {
        Eigen::Matrix<double, NU, 1> uk;
        uk << z(k);
        last_output.push_back(uk);
    }
    is_initialized_ = true;

    const auto t_total1 = std::chrono::steady_clock::now();

    // -------- accumulate & print every WINDOW_N calls --------
    const double ms_total   = ms_since(t_total0, t_total1);
    const double ms_Ac      = ms_since(t_Ac0,   t_Ac1);
    const double ms_expm    = ms_since(t_expm0, t_expm1);
    const double ms_w       = ms_since(t_w0,    t_w1);
    const double ms_Hg      = ms_since(t_H0,    t_H1);
    const double ms_llt     = ms_since(t_llt0,  t_llt1);
    const double ms_sol     = ms_since(t_sol0,  t_sol1);
    const double ms_pred    = ms_since(t_pr0,   t_pr1);

    st_total.add(ms_total);
    st_build_Ac.add(ms_Ac);
    st_expm.add(ms_expm);
    st_weights.add(ms_w);
    st_build_Hg.add(ms_Hg);
    st_llt.add(ms_llt);
    st_solve.add(ms_sol);
    st_predict.add(ms_pred);

    window_cnt++;

    if (window_cnt >= WINDOW_N) {
        const double inv = 1.0 / (double)window_cnt;

        ROS_INFO_STREAM(
            "[MPC TIMING] window=" << window_cnt
            << " | total avg=" << st_total.sum_ms*inv << " ms (max " << st_total.max_ms << ")"
            << " | Ac avg="    << st_build_Ac.sum_ms*inv << " (max " << st_build_Ac.max_ms << ")"
            << " | expm avg="  << st_expm.sum_ms*inv     << " (max " << st_expm.max_ms << ")"
            << " | W avg="     << st_weights.sum_ms*inv  << " (max " << st_weights.max_ms << ")"
            << " | Hg avg="    << st_build_Hg.sum_ms*inv << " (max " << st_build_Hg.max_ms << ")"
            << " | LLT avg="   << st_llt.sum_ms*inv      << " (max " << st_llt.max_ms << ")"
            << " | solve avg=" << st_solve.sum_ms*inv    << " (max " << st_solve.max_ms << ")"
            << " | pred avg="  << st_predict.sum_ms*inv  << " (max " << st_predict.max_ms << ")"
        );

        window_cnt = 0;
        st_total.reset();
        st_build_Ac.reset();
        st_expm.reset();
        st_weights.reset();
        st_build_Hg.reset();
        st_llt.reset();
        st_solve.reset();
        st_predict.reset();
    }

    return {ddelta_opt, mtv_opt, true, next_r};
}

MPC_Return MPCInterface::solve(const MPC_State& x0,
                               const Eigen::VectorXd& curvature_ref,
                               const Eigen::VectorXd& velocity_ref,
                               const Eigen::VectorXd& acceleration_ref,
                               double vx0_body)
{
    (void)acceleration_ref;
    (void)vx0_body;
    return solve(x0, curvature_ref, velocity_ref);
}

} // namespace v2_control