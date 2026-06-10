#pragma once
#include "ParamBank.hpp"
#include "uttilities.hpp"

// (+) omega clockwise – Twoja konwencja

namespace lem_dynamics_sim_ {

inline State derative_wheels_dynamics_model(const ParamBank& P, const State& x, const Input& u) {
    State temp; temp.setZero();

    const double dt = P.get("simulation_time_step");
    const double I  = P.get("I_tire");
    const double R  = P.get("R");

    double b = 0.0;

    auto omega_zoh_next = [&](double omega_k, double T, double Fx)->double {
        const double rhs = (T - Fx * R);       // [N·m]
        if (std::abs(b) > 1e-12) {
            const double z   = (b / I) * dt;
            const double a   = std::exp(-z);
            const double gain = (-std::expm1(-z)) / b; // = (1 - a)/b
            return a * omega_k + gain * rhs;
        } else {
            return omega_k + (dt / I) * rhs;
        }
    };

    // 4x4: Fx for every wheel
    const double omega_rr_next = omega_zoh_next(x.omega_rr, x.torque_rr, x.fx_rr);
    const double omega_rl_next = omega_zoh_next(x.omega_rl, x.torque_rl, x.fx_rl);
    const double omega_fr_next = omega_zoh_next(x.omega_fr, x.torque_fr, x.fx_fr);
    const double omega_fl_next = omega_zoh_next(x.omega_fl, x.torque_fl, x.fx_fl);

    // dω/dt = (ω_new - ω_old)/dt
    temp.omega_rr = (omega_rr_next - x.omega_rr) / dt;
    temp.omega_rl = (omega_rl_next - x.omega_rl) / dt;
    temp.omega_fr = (omega_fr_next - x.omega_fr) / dt;
    temp.omega_fl = (omega_fl_next - x.omega_fl) / dt;

    return temp;
}

} // namespace lem_dynamics_sim_
