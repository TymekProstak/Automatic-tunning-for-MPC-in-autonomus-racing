#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

#include <ros/ros.h>

#include "spline.hpp"       // TrackSpline2D  (ma signedNormalDistance, projectToSpline, eval, getCurvature, getYaw)
#include "ParamBank.hpp"
#include "Vec2.hpp"
#include "utilities.hpp"
#include "mpc_interface_agh_racing.hpp"

namespace v2_control {

// =====================================================
// Velocity planner structs
// =====================================================
struct SpeedProfileGeom
{
    double s0 = 0.0;     // start profilu (s na torze)
    double ds = 0.5;     // krok przestrzenny
    double S_plan = 0.0; // długość profilu [m] do przodu
    std::vector<double> v;
    std::vector<double> kappa;
    std::vector<double> a;
};

struct VelocityPlannerResult
{
    int N = 0;
    Eigen::VectorXd curvature;
    Eigen::VectorXd acceleration_ref;
    Eigen::VectorXd velocity_ref;
    Eigen::VectorXd X_ref;
    Eigen::VectorXd Y_ref;
    bool valid = false;
};

static inline double safeSqrt(double x) { return std::sqrt(std::max(0.0, x)); }

static inline double wrapS(double s, double L)
{
    if (L <= 1e-12) return 0.0;
    s = std::fmod(s, L);
    if (s < 0.0) s += L;
    return s;
}

// =====================================================
// Velocity planner: limits & vsat
// =====================================================

// friction ellipse longitudinal availability (WITH AERO DOWNFORCE)
static inline double longAvailAx(double v, double kappa, const ParamBank& P, bool acc = true)
{
    const double safety = P.get("vel_planner_saftey_factor");

    const double mux = P.get(acc ? "vel_planner_mux_acc" : "vel_planner_mux_dec") * safety;
    const double muy = P.get("vel_planner_muy") * safety;

    const double Cl  = P.get("vel_planner_Cl");
    const double m   = P.get("vel_planner_m");
    const double g   = 9.81;

    const double Fz = m*g + Cl*v*v;      // [N]
    const double ay = v*v*kappa;         // [m/s^2]

    const double ay_max = muy * (Fz / m);  // [m/s^2]
    const double ax_max = mux * (Fz / m);  // [m/s^2]

    if (ay_max <= 1e-12) return 0.0;

    const double ratio  = ay / ay_max;
    const double inside = 1.0 - ratio*ratio;

    return ax_max * std::sqrt(std::max(0.0, inside));
}

static inline void accelBoundsAt(
    double v_here, double k_here,
    double& a_min_out, double& a_max_out,
    const ParamBank& P)
{
    a_max_out =  longAvailAx(v_here, k_here, P, true);
    a_min_out = -longAvailAx(v_here, k_here, P, false);
}

static inline std::vector<double> buildVSat(
    const std::vector<double>& kappa,
    double v_max,
    const ParamBank& P)
{
    const int N = (int)kappa.size();
    std::vector<double> vsat(N, v_max);

    const double safety = P.get("vel_planner_saftey_factor");
    const double muy = P.get("vel_planner_muy") * safety;

    const double Cl  = P.get("vel_planner_Cl");
    const double m   = P.get("vel_planner_m");
    const double g   = 9.81;

    for (int i = 0; i < N; ++i) {
        const double kk = std::abs(kappa[i]);
        double v_kappa = v_max;

        if (kk > 1e-9) {
            // ay = v^2*kappa <= muy*(g + (Cl/m)*v^2)
            // => v^2*(kappa - muy*Cl/m) <= muy*g
            const double denom = kk - (muy * Cl / m);
            if (denom > 1e-6) {
                const double v2 = (muy * g) / denom;
                v_kappa = std::sqrt(std::max(0.0, v2));
            } else {
                v_kappa = v_max;
            }
        }

        vsat[i] = std::clamp(v_kappa, 0.0, v_max);
    }
    return vsat;
}

// forward: 3 cases (a_to_sat vs bounds)
static inline void forwardPass_threeCases(
    const std::vector<double>& kappa,
    const std::vector<double>& vsat,
    double ds,
    double v_min,
    double v_max,
    double v0,
    std::vector<double>& v,
    std::vector<double>& a,
    std::vector<bool>& is_valid,
    const ParamBank& P)
{
    const int N = (int)kappa.size();
    v.assign(N, 0.0);
    a.assign(N, 0.0);
    is_valid.assign(N, true);

    v[0] = v0;

    for (int i = 0; i < N - 1; ++i) {
        const double v0i = v[i];
        const double k0  = kappa[i];

        double a_av_min = 0.0, a_av_max = 0.0;
        accelBoundsAt(v0i, k0, a_av_min, a_av_max, P);

        const double v_sat_next = vsat[i+1];
        const double a_to_sat = (v_sat_next*v_sat_next - v0i*v0i) / (2.0*ds);

        if (a_to_sat >= a_av_max) {
            a[i] = a_av_max;
            v[i+1] = safeSqrt(v0i*v0i + 2.0*a[i]*ds);
            v[i+1] = std::min(v[i+1], v_sat_next);
            is_valid[i] = true;
        }
        else if (a_to_sat >= a_av_min && a_to_sat < a_av_max) {
            a[i] = a_to_sat;
            v[i+1] = v_sat_next;
            is_valid[i] = true;
        }
        else {
            a[i] = 0.0;
            v[i+1] = v_sat_next;
            is_valid[i] = false;
        }

        v[i+1] = std::clamp(v[i+1], v_min, v_max);
        v[i+1] = std::min(v[i+1], vsat[i+1]);
    }

    if (N >= 2) a[N-1] = a[N-2];
}

// backward: full fix using local bounds (ellipse)
static inline void fullBackwardPass_fix(
    const std::vector<double>& kappa,
    const std::vector<double>& vsat,
    double ds,
    std::vector<double>& v,
    std::vector<double>& a,
    std::vector<bool>& is_valid,
    const ParamBank& P)
{
    const int N = (int)kappa.size();
    if ((int)v.size() != N || (int)a.size() != N) return;

    for (int i = N - 1; i >= 2; --i) {
        const double k0 = kappa[i-1];
        const double k1 = kappa[i];

        const double v1 = v[i];
        const double v0_old = v[i-1];

        const double a_avg = (v1*v1 - v0_old*v0_old) / (2.0*ds);
        const bool a_matches = std::abs(a[i-1] - a_avg) < 1e-6;

        if (is_valid[i-1] && a_matches) continue;

        double a1_min = 0.0, a1_max = 0.0;
        accelBoundsAt(v1, k1, a1_min, a1_max, P);

        const double v0_reach_max = safeSqrt(v1*v1 - 2.0*a1_min*ds);
        const double v0_reach_min = safeSqrt(std::max(0.0, v1*v1 - 2.0*a1_max*ds));

        double a0_min = 0.0, a0_max = 0.0;
        accelBoundsAt(v0_old, k0, a0_min, a0_max, P);

        const bool valid_v = (v0_old >= v0_reach_min - 1e-9) && (v0_old <= v0_reach_max + 1e-9);
        const bool valid_a_end   = (a_avg >= a1_min - 1e-9) && (a_avg <= a1_max + 1e-9);
        const bool valid_a_start = (a_avg >= a0_min - 1e-9) && (a_avg <= a0_max + 1e-9);

        if (valid_v && valid_a_end && valid_a_start) {
            a[i-1] = a_avg;
            is_valid[i-1] = true;
            continue;
        }

        const double a_brake = a1_min; // <=0

        const double v0_max  = safeSqrt(v1*v1 - 2.0*a_brake*ds);
        v[i-1] = std::min(v[i-1], v0_max);

        a[i-1] = (v1*v1 - v[i-1]*v[i-1]) / (2.0*ds);
        is_valid[i-1] = true;
    }
    double a0 = (v[1]*v[1] - v[0]*v[0])/(2.0*ds);
    a0 = std::clamp(a0, -longAvailAx(v[0], kappa[0], P, false), longAvailAx(v[0], kappa[0], P, true));
    a[0] = a0;

    if (N >= 2) a[N-1] = a[N-2];
}

// jerk clamp: dt = 2*ds/(v0+v1)
static inline void jerkForwardClamp(
    const std::vector<double>& kappa,
    const std::vector<double>& vsat,
    double ds,
    double jerk_up,
    double jerk_down,
    double a0_along,
    std::vector<double>& v,
    std::vector<double>& a,
    std::vector<bool>& is_valid,
    const ParamBank& P)
{
    const int N = (int)kappa.size();
    if ((int)v.size() != N || (int)a.size() != N) return;

    const double j_up   = std::max(0.0, jerk_up);
    const double j_down = std::max(0.0, jerk_down);

   
    double a_prev = a0_along;
    // {
    //     double a_min0, a_max0;
    //     accelBoundsAt(v[0], kappa[0], a_min0, a_max0, P);
    //     a_prev = std::clamp(a_prev, a_min0, a_max0);
    // }


    for (int i = 0; i < N - 1; ++i) {
        const double v0i = v[i];
        const double k0  = kappa[i];

        double v1_guess = v[i+1];
        if (!std::isfinite(v1_guess)) v1_guess = v0i;

        const double v_sum = std::max(1e-3, v0i + v1_guess);
        const double dt = (2.0*ds) / v_sum;

        const double a_jmin = a_prev - j_down * dt;
        const double a_jmax = a_prev + j_up   * dt;

        double ai = a[i];
        if (!std::isfinite(ai)) ai = 0.0;

        ai = std::clamp(ai, a_jmin, a_jmax);

        double v1 = safeSqrt(v0i*v0i + 2.0*ai*ds);

        if (v1 > vsat[i+1]) {
            is_valid[i] = false;
            v1 = vsat[i+1];
            ai = (v1*v1 - v0i*v0i) / (2.0*ds);
        }

        a[i] = ai;
        v[i+1] = v1;
        a_prev = ai;
    }

    if (N >= 2) a[N-1] = a[N-2];
}

static inline void final_jerkForwardClamp(
    const std::vector<double>& kappa,
    const std::vector<double>& vsat,
    double ds,
    double jerk_up,
    double jerk_down,
    double a0_along,
    std::vector<double>& v,
    std::vector<double>& a,
    std::vector<bool>& is_valid,
    const ParamBank& P)
{
    const int N = (int)kappa.size();
    if ((int)v.size() != N || (int)a.size() != N) return;

    const double j_up   = std::max(0.0, jerk_up);
    const double j_down = std::max(0.0, jerk_down);

    double a_prev = a0_along;
    // {
    //     double a_min0, a_max0;
    //     accelBoundsAt(v[0], kappa[0], a_min0, a_max0, P);
    //     a_prev = std::clamp(a_prev, a_min0, a_max0);
    // }

    for (int i = 0; i < N - 1; ++i) {
        const double v0i = v[i];
        const double k0  = kappa[i];

        double a_min, a_max;
        accelBoundsAt(v0i, k0, a_min, a_max, P);

        double v1_guess = v[i+1];
        if (!std::isfinite(v1_guess)) v1_guess = v0i;

        const double v_sum = std::max(1e-3, v0i + v1_guess);
        const double dt = (2.0*ds) / v_sum;

        const double a_jmin = a_prev - j_down * dt;
        const double a_jmax = a_prev + j_up   * dt;

        double ai = a[i];
        if (!std::isfinite(ai)) ai = 0.0;

        ai = std::clamp(ai, a_jmin, a_jmax);

        double v1 = safeSqrt(v0i*v0i + 2.0*ai*ds);

        // if (v1 > vsat[i+1]) {
        //     is_valid[i] = false;
        //     v1 = vsat[i+1];
        //     ai = (v1*v1 - v0i*v0i) / (2.0*ds);
        // }

        a[i] = ai;
        v[i+1] = v1;
        a_prev = ai;
    }

    if (N >= 2) a[N-1] = a[N-2];
}

// main planner: forward/backward + (jerk clamp -> FULL backward)
static inline SpeedProfileGeom forward_backward_pass_with_jerk_full_backward(
    const TrackSpline2D& sp_closed,
    double s0,
    double ds_geom,
    double S_plan,
    double v0_along,
    double v_min,
    double v_max,
    double jerk_up,
    double jerk_down,
    double a0_along,
    int merge_iter,
    const ParamBank& P)
{
    SpeedProfileGeom prof;
    prof.s0 = s0;
    prof.ds = ds_geom;
    prof.S_plan = S_plan;

    const double L = sp_closed.totalLength();
    if (!sp_closed.valid() || !sp_closed.isClosed() || L <= 1e-6) return prof;
    if (ds_geom <= 1e-6 || S_plan <= ds_geom) return prof;

    const int N = (int)std::floor(S_plan / ds_geom) + 1;

    prof.v.assign(N, 0.0);
    prof.kappa.assign(N, 0.0);
    prof.a.assign(N, 0.0);

    for (int i = 0; i < N; ++i) {
        const double s = wrapS(s0 + (double)i * ds_geom, L);
        prof.kappa[i] = sp_closed.getCurvature(s);
    }

    const std::vector<double> vsat = buildVSat(prof.kappa, v_max, P);

    std::vector<double> v, a;
    std::vector<bool> is_valid;

    forwardPass_threeCases(prof.kappa, vsat, ds_geom, v_min, v_max, v0_along, v, a, is_valid, P);
    fullBackwardPass_fix(prof.kappa, vsat, ds_geom, v, a, is_valid, P);

    // for (int it = 0; it < std::max(0, merge_iter); ++it) {
    //     jerkForwardClamp(prof.kappa, vsat, ds_geom, jerk_up, jerk_down, a0_along, v, a, is_valid, P);
    //     fullBackwardPass_fix(prof.kappa, vsat, ds_geom, v, a, is_valid, P);
    // }

    // const double smooth_factor = std::max(1e-9, P.get("vel_planner_smoothing_factor"));
    // final_jerkForwardClamp(
    //     prof.kappa, vsat, ds_geom,
    //     jerk_up / smooth_factor,
    //     jerk_down / smooth_factor,
    //     a0_along, v, a, is_valid, P
    // );

    for (int i = 1; i < N; ++i) {
        double vi = v[i];
        if (!std::isfinite(vi)) vi = 0.0;
        vi = std::clamp(vi, v_min, v_max);
        //vi = std::min(vi, vsat[i]);
        v[i] = vi;

        if (!std::isfinite(a[i])) a[i] = 0.0;
    }

    prof.v = std::move(v);
    prof.a = std::move(a);

    return prof;
}

struct Profile_at_single_point
{
    double d0   = 0.0;  // odległość do przodu od prof.s0 [m]
    double a_ref = 0.0;
    double v_ref = 0.0;
    double k_ref = 0.0;
    double R_ref = 0.0;
};

// absolutne s_query -> odległość do przodu od s0 (wrapped), potem mapowanie na indeks profilu
static inline Profile_at_single_point profile_atS(
    const SpeedProfileGeom& prof,
    double s_query,
    double L_total)
{
    Profile_at_single_point res;

    const int N = (int)prof.v.size();
    if (N < 2) return res;

    const double d_fwd = wrapS(s_query - prof.s0, L_total);
    const double d = std::clamp(d_fwd, 0.0, std::max(0.0, prof.S_plan));
    res.d0 = d;

    const double u = d / prof.ds;
    int i = (int)std::floor(u);
    i = std::clamp(i, 0, N - 2);

    double alpha = u - (double)i;
    alpha = std::clamp(alpha, 0.0, 1.0);

    auto lerp = [&](double x0, double x1) -> double {
        return (1.0 - alpha)*x0 + alpha*x1;
    };

    res.v_ref = lerp(prof.v[i],     prof.v[i+1]);
    res.a_ref = lerp(prof.a[i],     prof.a[i+1]);
    res.k_ref = lerp(prof.kappa[i], prof.kappa[i+1]);

    const double kk = std::abs(res.k_ref);
    res.R_ref = (kk > 1e-9) ? (1.0 / kk) : std::numeric_limits<double>::infinity();

    return res;
}

// =====================================================
// NOWA WERSJA: planner NIE robi rzutu — dostaje s0_exact z zewnątrz
// =====================================================
inline VelocityPlannerResult velocity_planner_process_for_control(
    const ParamBank& P,
    const TrackSpline2D& spline_closed,
    const State& bolide_state,
    double s0_exact)
{
    VelocityPlannerResult out;

    if (!spline_closed.valid() || !spline_closed.isClosed()) {
        ROS_WARN_STREAM("[VelPlanner] spline invalid or not closed");
        return out;
    }

    const double L = spline_closed.totalLength();
    const int    K = (int)P.get("mpc_N");

    if (K < 2 || L <= 1e-6) {
        ROS_WARN_STREAM("[VelPlanner] bad K or L");
        return out;
    }

    const double ds_geom = P.get("vel_planner_spatial_step");
    if (ds_geom <= 1e-6) {
        ROS_WARN_STREAM("[VelPlanner] bad ds_geom");
        return out;
    }

    const double v_min = std::max(0.0, P.get("vel_planner_v_min"));
    const double v_max = std::max(v_min + 1e-3, P.get("vel_planner_v_max"));

    const double j_max = std::max(0.0, P.get("vel_planner_max_jerk"));
    const int merge_iter = (int)P.get("vel_planner_number_of_jerk_merging_iterations");

    const double mpc_dt = 1.0 / P.get("odom_frequency");

    // OKNO: 30 m do przodu (jeśli tor krótszy, to max L)
    const double S_plan = std::min(30.0, L);
    if (S_plan <= ds_geom) {
        ROS_WARN_STREAM("[VelPlanner] S_plan too small");
        return out;
    }

    // s0 jest “dokładnym rzutem” z zewnątrz
    double s0 = s0_exact;
    if (!std::isfinite(s0)) s0 = 0.0;
    s0 = wrapS(s0, L);

    // v0
    double v0_along = bolide_state.vx;
    if (!std::isfinite(v0_along)) v0_along = v_min;
    v0_along = std::max(v0_along, v_min);

    // a0 clamp do elipsy w punkcie startowym
    const double k0 = spline_closed.getCurvature(s0);
    double a0_min, a0_max;
    accelBoundsAt(v0_along, k0, a0_min, a0_max, P);

    double a0_along = bolide_state.acc;
    if (!std::isfinite(a0_along)) a0_along = 0.0;
    a0_along = std::clamp(a0_along, a0_min, a0_max);

    // LOCAL profile (tylko 30m)
    SpeedProfileGeom prof = forward_backward_pass_with_jerk_full_backward(
        spline_closed,
        s0,
        ds_geom,
        S_plan,
        v0_along,
        v_min,
        v_max,
        j_max,
        j_max,
        a0_along,
        merge_iter,
        P
    );

    if (prof.v.size() < 2) {
        ROS_WARN_STREAM("[VelPlanner] failed to build speed profile");
        return out;
    }

    // output refs for MPC horizon
    out.N = K;
    out.curvature.resize(K);
    out.velocity_ref.resize(K);
    out.acceleration_ref.resize(K);
    out.X_ref.resize(K);
    out.Y_ref.resize(K);

    out.valid = true;

    double s_along = s0;
    for (int i = 0; i < K; ++i) {
        s_along = wrapS(s_along, L);

        const auto pr = profile_atS(prof, s_along, L);

        out.curvature(i) = spline_closed.getCurvature(s_along);
        out.velocity_ref(i) = pr.v_ref;
        out.acceleration_ref(i) = pr.a_ref;

        Vec2 p = spline_closed.eval(s_along);
        out.X_ref(i) = p.x;
        out.Y_ref(i) = p.y;

        // integracja po czasie dla kolejnego punktu predykcji
        s_along += pr.v_ref * mpc_dt + 0.5 * pr.a_ref * mpc_dt * mpc_dt;
    }

    return out;
}

// =====================================================
// STARA WERSJA (kompatybilność): robi rzut wewnątrz (możesz NIE używać)
// =====================================================
inline VelocityPlannerResult velocity_planner_process_for_control(
    const ParamBank& P,
    const TrackSpline2D& spline_closed,
    const State& bolide_state)
{
    if (!spline_closed.valid() || !spline_closed.isClosed()) {
        VelocityPlannerResult out;
        return out;
    }

    const double L = spline_closed.totalLength();
    const Vec2 q((float)bolide_state.X, (float)bolide_state.Y);

    // szybki rzut (jeśli masz projectToSpline) — bez coarse scan
    double s0 = 0.0;
    try {
        s0 = spline_closed.projectToSpline(q);
    } catch (...) {
        s0 = 0.0;
    }
    s0 = wrapS(s0, L);

    return velocity_planner_process_for_control(P, spline_closed, bolide_state, s0);
}

} // namespace v2_control
