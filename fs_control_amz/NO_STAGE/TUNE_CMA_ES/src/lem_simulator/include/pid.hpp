#pragma once

#include <algorithm> // std::clamp
#include <cmath>     // std::isfinite, std::abs, std::exp
#include "uttilities.hpp"

namespace lem_dynamics_sim_ {

struct PIDParams {
    double Kp{0.0};
    double Ki{0.0};
    double Kd{0.0};

    double saturation_upper{0.0};
    double saturation_lower{0.0};

    double anti_windup_gain{0.0};
    double leak_time_scale{1.0};
};

class PIDController {
public:
    PIDController() = default;
    explicit PIDController(const PIDParams& params) : params_(params) {}

    void set_params(const PIDParams& params) { params_ = params; }

    void reset()
    {
        integrator_ = 0.0;
        prev_error_ = 0.0;
        output_     = 0.0;
        p_term_     = 0.0;
        i_term_     = 0.0;
        d_term_     = 0.0;
        active      = false;
    }

    void update(double error, double dt, bool on_off = true, bool leak_even_when_on = false)
    {
        active = on_off;

        // ---- HARD SAFETY: dt/error must be finite ----
        if (!std::isfinite(dt) || dt <= 1e-9) {
            // nie ruszam stanu (albo możesz resetować) – ważne: nie dzielić przez dt
            p_term_ = i_term_ = d_term_ = 0.0;
            output_ = 0.0;
            return;
        }

        if (!std::isfinite(error)) {
            // 1 tick NaN w sensorze = prev_error_ robi się NaN i masz NaN "wiecznie"
            // dlatego twardy reset
            reset();
            active = on_off;
            return;
        }

        auto leak_integrator = [&]() {
            const double tau = (params_.leak_time_scale > 1e-9) ? params_.leak_time_scale : 1e-9;
            const double alpha = std::exp(-dt / tau);   // stabilne numerycznie
            integrator_ *= alpha;
        };

        // ---- PID OFF -> tylko leak + zerowanie składowych ----
        if (!on_off) {
            leak_integrator();
            p_term_ = 0.0;
            i_term_ = 0.0;
            d_term_ = 0.0;
            output_ = 0.0;
            // opcjonalnie: prev_error_ = 0.0; (ja zostawiam, bo i tak OFF)
            return;
        }

        // ---- optional leak even when ON (np. near zero error) ----
        if (leak_even_when_on) {
            if (std::abs(error) < 1e-3) {
                leak_integrator();
            }
        }

        // 1) P
        p_term_ = params_.Kp * error;

        // 2) I (tylko jeśli Ki != 0)
        if (std::abs(params_.Ki) > 0.0) {
            integrator_ += error * dt;
            if (!std::isfinite(integrator_)) {
                reset();
                active = true;
                return;
            }
            i_term_ = params_.Ki * integrator_;
        } else {
            i_term_ = 0.0;
        }

        // 3) D (KLUCZ: nie licz pochodnej jeśli Kd==0)
        if (std::abs(params_.Kd) > 0.0) {
            const double derr = (error - prev_error_) / dt;
            d_term_ = params_.Kd * derr;
        } else {
            d_term_ = 0.0;
        }

        // 4) suma
        const double u_unsat = p_term_ + i_term_ + d_term_;
        if (!std::isfinite(u_unsat)) {
            reset();
            active = true;
            return;
        }

        // 5) clamp
        output_ = std::clamp(u_unsat, params_.saturation_lower, params_.saturation_upper);
        if (!std::isfinite(output_)) {
            reset();
            active = true;
            return;
        }

        // 6) anti-windup (tylko sensownie, jeśli Ki != 0)
        if (std::abs(params_.Ki) > 0.0 && std::abs(params_.anti_windup_gain) > 0.0) {
            const double u_error = output_ - u_unsat; // finite - finite
            if (std::isfinite(u_error)) {
                integrator_ += params_.anti_windup_gain * u_error * dt;
                if (!std::isfinite(integrator_)) {
                    reset();
                    active = true;
                    return;
                }
                // odśwież I-term do debug (po anti-windup)
                i_term_ = params_.Ki * integrator_;
            }
        }

        // zapamiętaj błąd na następny tick
        prev_error_ = error;
    }

    double get_output() const { return output_; }
    double get_P_term() const { return p_term_; }
    double get_I_term_integrator() const { return i_term_; }
    double get_D_term() const { return d_term_; }

    bool is_active() const { return active; }

private:
    PIDParams params_{};
    double integrator_{0.0};
    double prev_error_{0.0};
    double output_{0.0};

    double p_term_{0.0};
    double i_term_{0.0};
    double d_term_{0.0};

    bool active{false};
};

} // namespace lem_dynamics_sim_