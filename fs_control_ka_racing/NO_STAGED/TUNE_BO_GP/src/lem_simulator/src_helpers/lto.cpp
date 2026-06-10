#include "lto.hpp"

#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

namespace lto {


// ============================================================
//  Referencyjna trajektoria toru (N próbek w [0,L) bez duplikatu)
//  (wewnętrzne, tylko do solvera)
// ============================================================
struct SplineTrajectoryRef
{
    Eigen::VectorXd X_ref;
    Eigen::VectorXd Y_ref;
    Eigen::VectorXd yaw_ref;
    Eigen::VectorXd s_ref;       // 0, ds, ..., (N-1)ds
    Eigen::VectorXd kappa_ref;   // kappa na ref

    double ds = 0.0;
    double total_length = 0.0;   // L = N*ds

    int N_data() const { return static_cast<int>(s_ref.size()); }

    double wrap_s(double s) const
    {
        const double L = total_length;
        if (L <= 0.0) return s;
        s = std::fmod(s, L);
        if (s < 0.0) s += L;
        return s;
    }

    int idx_from_s(double s) const
    {
        const int N = N_data();
        if (N <= 0 || ds <= 0.0) return 0;
        const double sw = wrap_s(s);
        int idx = static_cast<int>(std::floor(sw / ds));
        idx %= N;
        if (idx < 0) idx += N;
        return idx;
    }

    double get_curvature_at_s(double s) const
    {
        const int N = N_data();
        if (N <= 0) return 0.0;
        return kappa_ref[idx_from_s(s)];
    }
};

// ============================================================
//  Geometria: yaw/normal/tangent na dyskretnej pętli
// ============================================================
static inline int wrap_i(int i, int N)
{
    int j = i % N;
    if (j < 0) j += N;
    return j;
}

static inline void tangent_normal_at_k(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    int k,
    double& yaw,
    double& nx,
    double& ny)
{
    const int N = (int)x.size();
    const int km1 = wrap_i(k - 1, N);
    const int kp1 = wrap_i(k + 1, N);

    const double dx = x[kp1] - x[km1];
    const double dy = y[kp1] - y[km1];

    yaw = std::atan2(dy, dx);

    // normal w lewo od kierunku
    nx = -std::sin(yaw);
    ny =  std::cos(yaw);
}

static inline void compute_yaw_closed_loop(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& yaw_out)
{
    const int N = (int)x.size();
    yaw_out.resize(N);
    for (int k = 0; k < N; ++k) {
        double yaw, nx, ny;
        tangent_normal_at_k(x, y, k, yaw, nx, ny);
        yaw_out[k] = yaw;
    }
}

// signed curvature from 3 points (closed loop)
static inline void compute_kappa_closed_loop(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& kappa_out)
{
    const int N = (int)x.size();
    kappa_out.resize(N);

    for (int k = 0; k < N; ++k) {
        const int km1 = wrap_i(k - 1, N);
        const int kp1 = wrap_i(k + 1, N);

        const double x1 = x[km1], y1 = y[km1];
        const double x2 = x[k],   y2 = y[k];
        const double x3 = x[kp1], y3 = y[kp1];

        const double ax = x2 - x1, ay = y2 - y1;
        const double bx = x3 - x2, by = y3 - y2;
        const double cx = x3 - x1, cy = y3 - y1;

        const double a = std::hypot(ax, ay);
        const double b = std::hypot(bx, by);
        const double c = std::hypot(cx, cy);

        // area2 = cross((p2-p1),(p3-p1)) = 2*Area (signed)
        const double area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);

        const double denom = a * b * c;
        if (denom < 1e-12 || std::abs(area2) < 1e-12) {
            kappa_out[k] = 0.0;
            continue;
        }

        // curvature = 4*Area/(abc) = 2*area2/(abc)
        kappa_out[k] = (2.0 * area2) / denom;
    }
}

// chord-length s for closed loop (no duplicate endpoint)
static inline void compute_s_chord_closed_loop(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    Eigen::VectorXd& s_out,
    double& total_len_out)
{
    const int N = (int)x.size();
    s_out.resize(N);
    s_out[0] = 0.0;

    double acc = 0.0;
    for (int k = 1; k < N; ++k) {
        const double dx = x[k] - x[k-1];
        const double dy = y[k] - y[k-1];
        acc += std::hypot(dx, dy);
        s_out[k] = acc;
    }

    // close the loop
    const double dx_last = x[0] - x[N-1];
    const double dy_last = y[0] - y[N-1];
    total_len_out = acc + std::hypot(dx_last, dy_last);
}

// ============================================================
//  Budowa ref traj z wejściowego XY (zamknięty spline)
// ============================================================
static SplineTrajectoryRef build_ref_from_xy(
    const Eigen::VectorXd& X_path,
    const Eigen::VectorXd& Y_path,
    const LtoParams& prm)
{
    if (X_path.size() != Y_path.size())
        throw std::runtime_error("LTO: X_path i Y_path mają różne rozmiary.");
    if (X_path.size() < 3)
        throw std::runtime_error("LTO: za mało punktów wejściowych (min 3).");
    if (prm.ds <= 0.0)
        throw std::runtime_error("LTO: prm.ds <= 0. Ustaw dodatni krok dyskretyzacji.");

    std::vector<v2_control::Vec2> pts;
    pts.reserve((size_t)X_path.size());
    for (int i = 0; i < X_path.size(); ++i)
        pts.emplace_back((float)X_path[i], (float)Y_path[i]);

    v2_control::TrackSpline2D spline;
    spline.build(pts, /*closed_loop=*/true);
    if (!spline.valid())
        throw std::runtime_error("LTO: TrackSpline2D: nie udało się zbudować poprawnego zamkniętego splajnu.");

    const double L = spline.totalLength();
    if (L <= 1e-6)
        throw std::runtime_error("LTO: spline totalLength() <= 0.");

    int N_data = (int)std::round(L / prm.ds);
    if (N_data < 3) N_data = 3;

    // Dostosowujemy ds, aby idealnie zamknąć pętlę bez gubienia reszty
    double actual_ds = L / (double)N_data;

    SplineTrajectoryRef traj;
    traj.ds = actual_ds;
    traj.total_length = L; // Teraz długość trajektorii = długość splajnu

    traj.X_ref.resize(N_data);
    traj.Y_ref.resize(N_data);
    traj.yaw_ref.resize(N_data);
    traj.s_ref.resize(N_data);
    traj.kappa_ref.resize(N_data);

    for (int i = 0; i < N_data; ++i) {
        const double s = (double)i * traj.ds;
        traj.s_ref[i]     = s;
        traj.X_ref[i]     = spline.getX(s);
        traj.Y_ref[i]     = spline.getY(s);
        traj.yaw_ref[i]   = spline.getYaw(s);
        traj.kappa_ref[i] = spline.getCurvature(s);
    }

    return traj;
}

// ============================================================
//  FG_eval (CppAD/Ipopt) - prywatne
// ============================================================
class FG_eval
{
public:
    using ADdouble = CppAD::AD<double>;
    using ADvector = CPPAD_TESTVECTOR(ADdouble);

private:
    static constexpr int NX = 7;
    static constexpr int NU = 3;  // [ddelta, dT, tv]

    // ineq:
    // track_left, track_right, ellipse_rear, ellipse_front, s_dot_min
    static constexpr int NY   = 5;
    static constexpr int NDYN = 7;

    static constexpr int N_PER   = 7;
    static constexpr int N_PHASE = 2;

private:
    SplineTrajectoryRef traj_;
    LtoParams prm_;

    int    N_steps_ = 0;   // N
    double ds_      = 0.0;

private:
    // z = [X0..XN, U0..U_{N-1}]
    static inline int idx_X_base(int k) { return k * NX; }
    static inline int idx_U_base(int k, int N) { return (N + 1) * NX + k * NU; }

    static inline int idx_n     (int k) { return idx_X_base(k) + 0; }
    static inline int idx_epsi  (int k) { return idx_X_base(k) + 1; }
    static inline int idx_vx    (int k) { return idx_X_base(k) + 2; }
    static inline int idx_vy    (int k) { return idx_X_base(k) + 3; }
    static inline int idx_r     (int k) { return idx_X_base(k) + 4; }
    static inline int idx_delta (int k) { return idx_X_base(k) + 5; }
    static inline int idx_T     (int k) { return idx_X_base(k) + 6; }

    static inline int idx_ddelta(int k, int N) { return idx_U_base(k, N) + 0; }
    static inline int idx_d_T   (int k, int N) { return idx_U_base(k, N) + 1; }
    static inline int idx_tv    (int k, int N) { return idx_U_base(k, N) + 2; }

private:
    ADdouble smooth_abs_ad(const ADdouble& x) const
    {
        const ADdouble eps = ADdouble(1e-4);
        return CppAD::sqrt(x*x + eps*eps);
    }

    ADdouble smooth_inv_ad(const ADdouble& x) const
    {
        const ADdouble eps = ADdouble(1e-3);
        return x / (x*x + eps*eps);
    }

    ADdouble compute_s_dot_ad(const ADvector& z, int k) const
    {
        const double s = double(k) * ds_;
        const double kappa_d = traj_.get_curvature_at_s(s);
        const ADdouble kappa = ADdouble(kappa_d);

        const ADdouble vx   = z[idx_vx(k)];
        const ADdouble vy   = z[idx_vy(k)];
        const ADdouble epsi = z[idx_epsi(k)];

        const ADdouble denom = (ADdouble(1.0) - kappa * z[idx_n(k)]);
        return (vx * CppAD::cos(epsi) - vy * CppAD::sin(epsi)) / denom;
    }

public:
    FG_eval(const Eigen::VectorXd& X_path,
            const Eigen::VectorXd& Y_path,
            const LtoParams& prm)
        : traj_(build_ref_from_xy(X_path, Y_path, prm))
        , prm_(prm)
    {
        N_steps_ = traj_.N_data();
        ds_      = traj_.ds;
    }

private:
    void build_initial_guess_(CPPAD_TESTVECTOR(double)& zi, int N) const
    {
        for (int i = 0; i < (int)zi.size(); ++i) zi[i] = 0.0;

        const double ds = ds_;
        const double Lwheel = prm_.lf + prm_.lr;

        const double ay_eff = 0.5 * prm_.mu_y * prm_.g;
        const double ax_eff = 0.5 * prm_.mu_x * prm_.g;

        auto clampd = [](double x, double a, double b) {
            return std::max(a, std::min(b, x));
        };

        std::vector<double> kappa_list(N, 0.0);
        for (int k = 0; k < N; ++k) {
            const double s = double(k) * ds;
            kappa_list[k] = traj_.get_curvature_at_s(s);
        }

        std::vector<double> v_lat(N, prm_.v_max);
        for (int k = 0; k < N; ++k) {
            const double kap = kappa_list[k];
            if (std::abs(kap) > 1e-9) {
                const double v = std::sqrt(std::max(0.0, ay_eff / std::abs(kap)));
                v_lat[k] = clampd(v, prm_.v_min, prm_.v_max);
            } else {
                v_lat[k] = prm_.v_max;
            }
        }

        std::vector<double> v_guess(N + 1, prm_.v_min);
        for (int k = 0; k < N; ++k) v_guess[k] = v_lat[k];
        v_guess[N] = v_guess[0];

        auto ax_available = [&](double v, double kap) -> double {
            const double ay = v * v * std::abs(kap);
            const double r  = ay_eff > 1e-12 ? (ay / ay_eff) : 1e9;
            const double inside = std::max(0.0, 1.0 - r * r);
            const double ax = ax_eff * std::sqrt(inside);
            return std::max(0.0, ax);
        };

        const int sweeps = 4;
        for (int it = 0; it < sweeps; ++it)
        {
            // forward
            for (int k = 0; k < N; ++k) {
                const int kp1 = (k + 1) % N;
                const double vk   = v_guess[k];
                const double kapk = kappa_list[k];
                const double ax = ax_available(vk, kapk);
                const double v2_next = std::max(0.0, vk * vk + 2.0 * ax * ds);
                const double v_next  = std::sqrt(v2_next);
                v_guess[k + 1] = std::min(v_lat[kp1], clampd(v_next, prm_.v_min, prm_.v_max));
            }
            v_guess[N] = v_guess[0];

            // backward
            for (int k = N - 1; k >= 0; --k) {
                const int kp1 = (k + 1) % N;
                const double vkp1 = v_guess[k + 1];
                const double kap1 = kappa_list[kp1];
                const double axb = ax_available(vkp1, kap1);
                const double v2_prev = std::max(0.0, vkp1 * vkp1 + 2.0 * axb * ds);
                const double v_prev  = std::sqrt(v2_prev);
                v_guess[k] = std::min(v_guess[k], clampd(v_prev, prm_.v_min, prm_.v_max));
            }
            v_guess[N] = v_guess[0];
        }

        // delta/beta
        std::vector<double> delta_guess(N + 1, 0.0);
        std::vector<double> beta_guess(N + 1, 0.0);

        for (int k = 0; k <= N; ++k) {
            const int kk = (k == N) ? 0 : k;
            const double kap = kappa_list[kk];

            double delta = std::atan(Lwheel * kap);
            delta = clampd(delta, prm_.min_delta, prm_.max_delta);
            delta_guess[k] = delta;

            const double beta = std::atan((prm_.lr / Lwheel) * std::tan(delta));
            beta_guess[k] = beta;
        }

        // states
        for (int k = 0; k <= N; ++k) {
            const int kk = (k == N) ? 0 : k;
            const double kap = kappa_list[kk];

            const double vx = v_guess[k];
            const double vy = vx * std::tan(beta_guess[k]);
            const double r  = vx * kap;

            zi[idx_n(k)]     = 0.0;
            zi[idx_epsi(k)]  = 0.0;
            zi[idx_vx(k)]    = vx;
            zi[idx_vy(k)]    = vy;
            zi[idx_r(k)]     = r;
            zi[idx_delta(k)] = delta_guess[k];
            zi[idx_T(k)]     = 0.0;
        }

        // throttle guess from a ~ dv^2/(2ds)
        std::vector<double> T_guess(N + 1, 0.0);
        for (int k = 0; k < N; ++k) {
            const double v0 = v_guess[k];
            const double v1 = v_guess[k + 1];

            const double a = (v1*v1 - v0*v0) / (2.0 * ds);
            const double Fx_req = prm_.m * a;

            double T = (std::abs(prm_.Cm) > 1e-9) ? (Fx_req /(2.0 * prm_.Cm)) : 0.0;
            T = clampd(T, -1.0, 1.0);

            T_guess[k] = T;
            zi[idx_T(k)] = T;
        }
        T_guess[N]   = T_guess[0];
        zi[idx_T(N)] = zi[idx_T(0)];

        // controls
        for (int k = 0; k < N; ++k) {
            const double s_dot_guess = std::max(prm_.s_dot_guess_floor, v_guess[k]);
            const double dt = ds / s_dot_guess;

            const double ddelta = (delta_guess[k+1] - delta_guess[k]) / dt;
            const double dT     = (T_guess[k+1]     - T_guess[k])     / dt;

            zi[idx_ddelta(k, N)] = clampd(ddelta, prm_.min_d_delta, prm_.max_d_delta);
            zi[idx_d_T(k, N)]    = clampd(dT,     prm_.min_d_T,     prm_.max_d_T);
            zi[idx_tv(k, N)]     = 0.0;
        }
    }

public:
    void operator()(ADvector& fg, const ADvector& z)
    {
        const int N = N_steps_;
        const double ds = ds_;

        fg[0] = ADdouble(0.0);

        for (int k = 0; k < N; ++k)
        {
            const ADdouble s_dot = compute_s_dot_ad(z, k);

            fg[0] += prm_.s_dot_cost * s_dot;

            const ADdouble ddelta = z[idx_ddelta(k, N)];
            const ADdouble dT     = z[idx_d_T(k, N)];
            const ADdouble tv     = z[idx_tv(k, N)];

            fg[0] += prm_.d_delta_cost * ddelta * ddelta;
            fg[0] += prm_.d_T_cost     * dT * dT;
            fg[0] += prm_.tv_cost      * tv * tv;

            const ADdouble beta_dyn = CppAD::atan2(z[idx_vy(k)], z[idx_vx(k)]);
            const ADdouble beta_kin = CppAD::atan(prm_.lr * z[idx_delta(k)] / (prm_.lf + prm_.lr));
            const ADdouble beta_err = beta_dyn - beta_kin;

            fg[0] += prm_.beta_cost * beta_err * beta_err;
        }

        int c = 1;

        for (int k = 0; k < N; ++k)
        {
            const ADdouble s_dot = compute_s_dot_ad(z, k);
            const ADdouble dt    = ADdouble(ds) * smooth_inv_ad(s_dot);

            fg[c++] = z[idx_n(k+1)] - z[idx_n(k)]
                    - (z[idx_vx(k)] * CppAD::sin(z[idx_epsi(k)]) +
                       z[idx_vy(k)] * CppAD::cos(z[idx_epsi(k)])) * dt;

            const double kappa_d = traj_.get_curvature_at_s(double(k) * ds);
            const ADdouble kappa = ADdouble(kappa_d);
            fg[c++] = z[idx_epsi(k+1)] - z[idx_epsi(k)]
                    - (z[idx_r(k)] - kappa * s_dot) * dt;

            const ADdouble alpha_f = CppAD::atan2(z[idx_vy(k)] + prm_.lf * z[idx_r(k)], z[idx_vx(k)]) - z[idx_delta(k)];
            const ADdouble alpha_r = CppAD::atan2(z[idx_vy(k)] - prm_.lr * z[idx_r(k)], z[idx_vx(k)]);

            const ADdouble Fz   = prm_.m * prm_.g + prm_.Cl * z[idx_vx(k)] * z[idx_vx(k)];
            const ADdouble Fz_f = Fz * prm_.lr / (prm_.lf + prm_.lr);
            const ADdouble Fz_r = Fz * prm_.lf / (prm_.lf + prm_.lr);

            const ADdouble Fy_f = Fz_f * prm_.D * CppAD::sin(prm_.C * CppAD::atan(prm_.B * alpha_f));
            const ADdouble Fy_r = Fz_r * prm_.D * CppAD::sin(prm_.C * CppAD::atan(prm_.B * alpha_r));

            const ADdouble F_motor = prm_.Cm * z[idx_T(k)];
            const ADdouble Fx_f    = ADdouble(0.5) * F_motor;
            const ADdouble Fx_r    = ADdouble(0.5) * F_motor;
            const ADdouble F_fric  = prm_.Cr0 + prm_.Cd *  z[idx_vx(k)] * z[idx_vx(k)];
            const ADdouble M_tv    = z[idx_tv(k, N)];

            fg[c++] = z[idx_vx(k+1)] - z[idx_vx(k)]
                    - (ADdouble(1.0)/prm_.m) *
                      (Fx_f + Fx_r * CppAD::cos(z[idx_delta(k)]) - F_fric
                       + Fy_f * CppAD::sin(z[idx_delta(k)])
                       + prm_.m * z[idx_r(k)] * z[idx_vy(k)]) * dt;

            fg[c++] = z[idx_vy(k+1)] - z[idx_vy(k)]
                    - (ADdouble(1.0)/prm_.m) *
                      (Fy_r + Fx_f * CppAD::sin(z[idx_delta(k)]) 
                       + Fy_f * CppAD::cos(z[idx_delta(k)])
                       - prm_.m * z[idx_r(k)] * z[idx_vx(k)]) * dt;

            fg[c++] = z[idx_r(k+1)] - z[idx_r(k)]
                    - (ADdouble(1.0)/prm_.Iz) *
                      (prm_.lf * (Fy_f * CppAD::cos(z[idx_delta(k)]) + Fx_f * CppAD::sin(z[idx_delta(k)]))
                       - prm_.lr * Fy_r
                       + M_tv) * dt;

            fg[c++] = z[idx_delta(k+1)] - z[idx_delta(k)]
                    - z[idx_ddelta(k, N)] * dt;

            fg[c++] = z[idx_T(k+1)] - z[idx_T(k)]
                    - z[idx_d_T(k, N)] * dt;
        }

        for (int k = 0; k < N; ++k)
        {
            const ADdouble n    = z[idx_n(k)];
            const ADdouble epsi = z[idx_epsi(k)];
            const ADdouble abs_epsi = smooth_abs_ad(epsi);

            fg[c++] = n
                    - (prm_.length/2.0) * CppAD::sin(abs_epsi)
                    + (prm_.width/2.0) * CppAD::cos(epsi)
                    - (prm_.track_width/2.0);

            fg[c++] = -n
                    + (prm_.length/2.0) * CppAD::sin(abs_epsi)
                    + (prm_.width/2.0) * CppAD::cos(epsi)
                    - (prm_.track_width/2.0);

            const ADdouble F_motor = prm_.Cm * z[idx_T(k)];
            const ADdouble Fx_f    = ADdouble(0.5) * F_motor;
            const ADdouble Fx_r    = ADdouble(0.5) * F_motor;

            const ADdouble alpha_f = CppAD::atan2(z[idx_vy(k)] + prm_.lf * z[idx_r(k)], z[idx_vx(k)]) - z[idx_delta(k)];
            const ADdouble alpha_r = CppAD::atan2(z[idx_vy(k)] - prm_.lr * z[idx_r(k)], z[idx_vx(k)]);

            const ADdouble Fz   = prm_.m * prm_.g + prm_.Cl * z[idx_vx(k)] * z[idx_vx(k)];
            const ADdouble Fz_f = Fz * prm_.lr / (prm_.lf + prm_.lr);
            const ADdouble Fz_r = Fz * prm_.lf / (prm_.lf + prm_.lr);

            const ADdouble Fy_f = Fz_f * prm_.D * CppAD::sin(prm_.C * CppAD::atan(prm_.B * alpha_f));
            const ADdouble Fy_r = Fz_r * prm_.D * CppAD::sin(prm_.C * CppAD::atan(prm_.B * alpha_r));

            fg[c++] =
                (Fx_r*Fx_r)/(Fz_r*Fz_r*prm_.mu_x*prm_.mu_x)
              + (Fy_r*Fy_r)/(Fz_r*Fz_r*prm_.mu_y*prm_.mu_y)
              - ADdouble(1.0);

            fg[c++] =
                (Fx_f*Fx_f)/(Fz_f*Fz_f*prm_.mu_x*prm_.mu_x)
              + (Fy_f*Fy_f)/(Fz_f*Fz_f*prm_.mu_y*prm_.mu_y)
              - ADdouble(1.0);

            const ADdouble s_dot = compute_s_dot_ad(z, k);
            fg[c++] = ADdouble(prm_.s_dot_min) - s_dot;
        }

        fg[c++] = z[idx_n(N)]     - z[idx_n(0)];
        fg[c++] = z[idx_epsi(N)]  - z[idx_epsi(0)];
        fg[c++] = z[idx_vx(N)]    - z[idx_vx(0)];
        fg[c++] = z[idx_vy(N)]    - z[idx_vy(0)];
        fg[c++] = z[idx_r(N)]     - z[idx_r(0)];
        fg[c++] = z[idx_delta(N)] - z[idx_delta(0)];
        fg[c++] = z[idx_T(N)]     - z[idx_T(0)];

        fg[c++] = z[idx_n(0)];
        fg[c++] = z[idx_epsi(0)];
    }

public:
    friend LtoResult solve_from_fg(FG_eval& fg_eval);
};

// ============================================================
//  Solve helper  (NOT static, because friend)
// ============================================================
LtoResult solve_from_fg(FG_eval& fg_eval)
{
    using Dvector = CPPAD_TESTVECTOR(double);

    const int N = fg_eval.N_steps_;
    if (N < 3) throw std::runtime_error("LTO: traj ma za mało punktów (N<3).");

    constexpr int NX = FG_eval::NX;
    constexpr int NU = FG_eval::NU;
    constexpr int NDYN = FG_eval::NDYN;
    constexpr int NY = FG_eval::NY;
    constexpr int N_PER = FG_eval::N_PER;
    constexpr int N_PHASE = FG_eval::N_PHASE;

    const int n_vars = (N + 1) * NX + N * NU;
    const int n_cons = NDYN * N + NY * N + N_PER + N_PHASE;

    Dvector xl(n_vars), xu(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        xl[i] = -std::numeric_limits<double>::infinity();
        xu[i] =  std::numeric_limits<double>::infinity();
    }

    const double epsi_max = M_PI / 4.0;

    for (int k = 0; k <= N; ++k) {
        xl[FG_eval::idx_vx(k)] = fg_eval.prm_.v_min;
        xu[FG_eval::idx_vx(k)] = fg_eval.prm_.v_max;

        xl[FG_eval::idx_delta(k)] = fg_eval.prm_.min_delta;
        xu[FG_eval::idx_delta(k)] = fg_eval.prm_.max_delta;

        xl[FG_eval::idx_T(k)] = -1.0;
        xu[FG_eval::idx_T(k)] =  1.0;

        xl[FG_eval::idx_epsi(k)] = -epsi_max;
        xu[FG_eval::idx_epsi(k)] =  epsi_max;
    }

    for (int k = 0; k < N; ++k) {
        xl[FG_eval::idx_ddelta(k, N)] = fg_eval.prm_.min_d_delta;
        xu[FG_eval::idx_ddelta(k, N)] = fg_eval.prm_.max_d_delta;

        xl[FG_eval::idx_d_T(k, N)]    = fg_eval.prm_.min_d_T;
        xu[FG_eval::idx_d_T(k, N)]    = fg_eval.prm_.max_d_T;

        xl[FG_eval::idx_tv(k, N)]     = fg_eval.prm_.min_tv;
        xu[FG_eval::idx_tv(k, N)]     = fg_eval.prm_.max_tv;
    }

    Dvector gl(n_cons), gu(n_cons);
    int c = 0;

    for (int k = 0; k < N; ++k)
        for (int i = 0; i < NDYN; ++i) { gl[c] = 0.0; gu[c] = 0.0; ++c; }

    for (int k = 0; k < N; ++k)
        for (int i = 0; i < NY; ++i) { gl[c] = -std::numeric_limits<double>::infinity(); gu[c] = 0.0; ++c; }

    for (int i = 0; i < N_PER;   ++i) { gl[c] = 0.0; gu[c] = 0.0; ++c; }
    for (int i = 0; i < N_PHASE; ++i) { gl[c] = 0.0; gu[c] = 0.0; ++c; }

    Dvector zi(n_vars);
    fg_eval.build_initial_guess_(zi, N);

    std::string options;
    // Opcjonalnie zredukuj print_level na 3 lub 4, żeby konsola nie spowalniała obliczeń
    options += "Integer print_level  4\n"; 
    options += "String  sb           yes\n";
    options += "String linear_solver mumps\n";

    // --- GŁÓWNE KRYTERIA ---
    options += "Numeric tol               5e-3\n"; 
    options += "Numeric dual_inf_tol      1e-2\n";
    // Poluzowane o rząd wielkości (z 1e-5 na 1e-4):
    options += "Numeric constr_viol_tol   1e-4\n"; 
    options += "Numeric compl_inf_tol     1e-4\n";

    // --- KRYTERIA RATUNKOWE (ACCEPTABLE) ---
    // Muszą być luźniejsze niż powyższe!
    options += "Numeric acceptable_tol               1e-2\n"; // 1e-2 jest większe od 5e-3
    options += "Integer acceptable_iter              8\n";
    options += "Numeric acceptable_dual_inf_tol      5e-2\n"; 
    options += "Numeric acceptable_constr_viol_tol   1e-3\n"; // 1e-3 jest większe od 1e-4
    options += "Numeric acceptable_compl_inf_tol     1e-2\n";

    options += "Numeric mu_init 1e-2\n";
    options += "String  mu_strategy adaptive\n";
    options += "String  line_search_method filter\n";

    options += "String  nlp_scaling_method gradient-based\n";
    options += "Numeric bound_push 1e-3\n";
    options += "Numeric bound_frac 1e-3\n";

    options += "String warm_start_init_point no\n";
    options += "Integer max_iter 3000\n";
    
    // Ustawione na docelowe 30 sekund
    options += "Numeric max_cpu_time 30\n"; 
    options += "Sparse true forward\n";
    CppAD::ipopt::solve_result<Dvector> sol;
    CppAD::ipopt::solve<Dvector, FG_eval>(options, zi, xl, xu, gl, gu, fg_eval, sol);

    LtoResult out;
    out.states.resize(N + 1);
    out.actions.resize(N);
    out.vx_list.resize(N);

    out.ipopt_status  = (int)sol.status;
    out.ipopt_success = (sol.status == CppAD::ipopt::solve_result<Dvector>::success);

    auto safe = [](double d){
        if (std::abs(d) < 1e-12) return (d >= 0 ? 1e-12 : -1e-12);
        return d;
    };

    for (int k = 0; k <= N; ++k) {
        MPC_State_LTO st;
        st.ey       = sol.x[FG_eval::idx_n(k)];
        st.epsi     = sol.x[FG_eval::idx_epsi(k)];
        st.vx       = sol.x[FG_eval::idx_vx(k)];
        st.vy       = sol.x[FG_eval::idx_vy(k)];
        st.r        = sol.x[FG_eval::idx_r(k)];
        st.delta    = sol.x[FG_eval::idx_delta(k)];
        st.throthle = sol.x[FG_eval::idx_T(k)];

        st.s        = double(k) * fg_eval.ds_;
        st.kappa    = fg_eval.traj_.get_curvature_at_s(st.s);

        const double denom = safe(1.0 - st.kappa * st.ey);
        st.s_dot = (st.vx * std::cos(st.epsi) - st.vy * std::sin(st.epsi)) / denom;

        out.states[k] = st;
        if (k < N) out.vx_list[k] = st.vx;
    }

    for (int k = 0; k < N; ++k) {
        MPC_Action_LTO ac;
        ac.ddelta_opt = sol.x[FG_eval::idx_ddelta(k, N)];
        ac.dthot_opt  = sol.x[FG_eval::idx_d_T(k, N)];
        ac.tv_opt     = sol.x[FG_eval::idx_tv(k, N)];
        out.actions[k] = ac;
    }

    out.x_opt.resize(N);
    out.y_opt.resize(N);
    out.lateral_deviation.resize((size_t)N);

    for (int k = 0; k < N; ++k) {
        double yaw_ref, nx, ny;
        tangent_normal_at_k(fg_eval.traj_.X_ref, fg_eval.traj_.Y_ref, k, yaw_ref, nx, ny);

        const double ey = out.states[k].ey;
        out.lateral_deviation[(size_t)k] = ey;

        out.x_opt[k] = fg_eval.traj_.X_ref[k] + ey * nx;
        out.y_opt[k] = fg_eval.traj_.Y_ref[k] + ey * ny;
    }

    compute_yaw_closed_loop(out.x_opt, out.y_opt, out.yaw_opt);
    compute_kappa_closed_loop(out.x_opt, out.y_opt, out.kappa_opt);
    compute_s_chord_closed_loop(out.x_opt, out.y_opt, out.s_opt, out.total_length_opt);

    return out;
}

// ============================================================
//  ParamBank -> LtoParams
// ============================================================
void LtoParams::load_lto_param_from_param_bank(const ParamBank& P)
{
    g     = P.get("lto_g");
    v_min = P.get("lto_v_min");
    v_max = P.get("lto_v_max");
    lf    = P.get("lto_lf");
    lr    = P.get("lto_lr");

    Cm  = P.get("lto_Cm");
    Cr0 = P.get("lto_Cr0");
    Cl  = P.get("lto_Cl");
    Cd = P.get("lto_Cd");

    max_drive_power = P.get("lto_max_drive_power");
    max_brake_power = P.get("lto_max_brake_power");

    max_delta = P.get("lto_max_delta");
    min_delta = P.get("lto_min_delta");

    max_d_delta = P.get("lto_max_d_delta");
    min_d_delta = P.get("lto_min_d_delta");

    max_d_T = P.get("lto_max_d_T");
    min_d_T = P.get("lto_min_d_T");

    max_tv = P.get("lto_max_tv");
    min_tv = P.get("lto_min_tv");

    Fz_nom = P.get("lto_Fz_nom");

    mu_y = P.get("lto_mu_y") * P.get("lto_saftey_factor");
    mu_x = P.get("lto_mu_x") * P.get("lto_saftey_factor");

    C = P.get("lto_C");
    D = P.get("lto_D");
    B = P.get("lto_B");

    length      = P.get("lto_length");
    width       = P.get("lto_width");
    track_width = P.get("lto_track_width");

    d_delta_cost = P.get("lto_d_delta_cost");
    d_T_cost     = P.get("lto_d_T_cost");
    beta_cost    = P.get("lto_beta_cost");
    s_dot_cost   = P.get("lto_s_dot_cost");
    tv_cost      = P.get("lto_tv_cost");

    m  = P.get("lto_m");
    Iz = P.get("lto_Iz");

    ds = P.get("lto_ds");
    if (ds <= 0.0) ds = 0.5;
}

// ============================================================
//  Public API
// ============================================================
LtoResult solve_lto_speed_profile(
    const Eigen::VectorXd& X_path,
    const Eigen::VectorXd& Y_path,
    const ParamBank& P)
{
    LtoParams prm;
    prm.load_lto_param_from_param_bank(P);
    return solve_lto_speed_profile(X_path, Y_path, prm);
}

LtoResult solve_lto_speed_profile(
    const Eigen::VectorXd& X_path,
    const Eigen::VectorXd& Y_path,
    const LtoParams& prm)
{
    FG_eval fg_eval(X_path, Y_path, prm);
    return solve_from_fg(fg_eval);
}

} // namespace lto