#include "tire_model.hpp"
#include <algorithm>
#include <cmath>

namespace lem_dynamics_sim_{

State derative_tire_model(const ParamBank& P, const State& x, const Input& u)
{
    State temp; temp.setZero();

    // ============================================================
    // 1) Parametry pojazdu
    // ============================================================
    const double m = P.get("m");
    const double g = P.get("g");
    const double w = P.get("w");
    const double a = P.get("a");
    const double b = P.get("b");
    const double t_front = P.get("t_front");
    const double t_rear  = P.get("t_rear");
    const double h = P.get("h");
    const double h_roll_f = P.get("h1_roll");
    const double h_roll_r = P.get("h2_roll");

    const double Kf = P.get("K1");
    const double Kr = P.get("K2");
    const double K_total = Kf + Kr;

    const double mf = m * a / w;
    const double mr = m * b / w;
    const double h_prim_f = h - h_roll_f;
    const double h_prim_r = h - h_roll_r;

    // ============================================================
    // 2) Parametry “numerical safety” + relaksacje
    // ============================================================
    const double epsilon = P.get("epsilon");
    const double cx = P.get("cx");
    const double dx = P.get("dx");
    const double cy = P.get("cy");
    const double dy = P.get("dy");

    const double dt = P.get("simulation_time_step");

    // ============================================================
    // 3) Geometria kół + promień
    // ============================================================
    const double r_rear  = P.get("r_rear");
    const double r_front = P.get("r_front");
    const double ang_f = P.get("angle_construction_front");
    const double ang_r = P.get("angle_construction_rear");
    const double R = P.get("R");

    // ============================================================
    // 4) MF 6.1 - parametry (no camber)
    // ============================================================

    // -------- LONGITUDINAL MF6.1 --------
    const double pCx1 = P.get("pCx1");

    const double pDx1 = P.get("pDx1");
    const double pDx2 = P.get("pDx2");

    const double pEx1 = P.get("pEx1");
    const double pEx2 = P.get("pEx2");
    const double pEx3 = P.get("pEx3");
    const double pEx4 = P.get("pEx4");

    const double pKx1 = P.get("pKx1");
    const double pKx2 = P.get("pKx2");
    const double pKx3 = P.get("pKx3");

    const double pHx1 = P.get("pHx1");
    const double pHx2 = P.get("pHx2");

    const double pVx1 = P.get("pVx1");
    const double pVx2 = P.get("pVx2");

    const double lambda_x = P.get("lambda_x");

    // -------- LATERAL MF6.1 --------
    const double pCy1 = P.get("pCy1");

    const double pDy1 = P.get("pDy1");
    const double pDy2 = P.get("pDy2");

    const double pEy1 = P.get("pEy1");
    const double pEy2 = P.get("pEy2");
    const double pEy3 = P.get("pEy3");

    const double pKy1 = P.get("pKy1");
    const double pKy2 = P.get("pKy2");
    const double pKy4 = P.get("pKy4");

    const double pHy1 = P.get("pHy1");
    const double pHy2 = P.get("pHy2");

    const double pVy1 = P.get("pVy1");
    const double pVy2 = P.get("pVy2");

    const double lambda_y = P.get("lambda_y");

    const double N0 = P.get("N0");

    double vx_rr = x.vx + x.yaw_rate * r_rear  * std::sin(ang_r);
    double vy_rr = x.vy - x.yaw_rate * r_rear  * std::cos(ang_r);

    double vx_rl = x.vx - x.yaw_rate * r_rear  * std::sin(ang_r);
    double vy_rl = x.vy - x.yaw_rate * r_rear  * std::cos(ang_r);

    double vx_fr = x.vx + x.yaw_rate * r_front * std::sin(ang_f);
    double vy_fr = x.vy + x.yaw_rate * r_front * std::cos(ang_f);

    double vx_fl = x.vx - x.yaw_rate * r_front * std::sin(ang_f);
    double vy_fl = x.vy + x.yaw_rate * r_front * std::cos(ang_f);

    // ============================================================
    // RZUT DO RAMY KOŁA
    // po tym bloku:
    //   vx_* = v_long
    //   vy_* = v_lat
    // ============================================================

    // RR
    {
        const double c = 1.0, s = 0.0;
        const double vx0 = vx_rr, vy0 = vy_rr;
        vx_rr =  c * vx0 + s * vy0;
        vy_rr = -s * vx0 + c * vy0;
    }

    // RL
    {
        const double c = 1.0, s = 0.0;
        const double vx0 = vx_rl, vy0 = vy_rl;
        vx_rl =  c * vx0 + s * vy0;
        vy_rl = -s * vx0 + c * vy0;
    }

    // FR
    {
        const double c = std::cos(x.delta_right);
        const double s = std::sin(x.delta_right);
        const double vx0 = vx_fr, vy0 = vy_fr;
        vx_fr =  c * vx0 + s * vy0;
        vy_fr = -s * vx0 + c * vy0;
    }

    // FL
    {
        const double c = std::cos(x.delta_left);
        const double s = std::sin(x.delta_left);
        const double vx0 = vx_fl, vy0 = vy_fl;
        vx_fl =  c * vx0 + s * vy0;
        vy_fl = -s * vx0 + c * vy0;
    }

    // ============================================================
    // 5) Clampy / epsy
    // ============================================================
    const double EPS_VX_ALPHA = std::max(epsilon, 1e-4);
    const double EPS_VX_KAPPA = std::max(epsilon, 1e-4);
    const double V_RELAX_MIN  = 1.0;
    const double L_RELAX_MIN  = 0.01;
    const double Z_MAX        = 50.0;

    auto safe_vx_signed = [&](double vx)->double {
        if (std::abs(vx) >= EPS_VX_ALPHA) return vx;
        return std::copysign(EPS_VX_ALPHA, (vx == 0.0 ? 1.0 : vx));
    };

    auto speed_mag = [&](double vx)->double {
        return std::max(std::abs(vx), V_RELAX_MIN);
    };

    auto vx_denom_kappa = [&](double vx, double omegaR){
        return std::max({std::abs(vx), std::abs(omegaR), EPS_VX_KAPPA});
    };

    const double vx_rr_den_kappa = vx_denom_kappa(vx_rr, x.omega_rr * R);
    const double vx_rl_den_kappa = vx_denom_kappa(vx_rl, x.omega_rl * R);
    const double vx_fr_den_kappa = vx_denom_kappa(vx_fr, x.omega_fr * R);
    const double vx_fl_den_kappa = vx_denom_kappa(vx_fl, x.omega_fl * R);

    // ============================================================
    // 6) Obciążenia normalne
    // ============================================================
    double N_fl = 0.5 * mf * g - 0.5 * m * x.prev_ax * h / w
                - x.prev_ay / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P.get("Cl1") * x.vx * x.vx;

    double N_fr = 0.5 * mf * g - 0.5 * m * x.prev_ax * h / w
                + x.prev_ay / t_front * ( mf * h_roll_f + Kf / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P.get("Cl1") * x.vx * x.vx;

    double N_rl = 0.5 * mr * g + 0.5 * m * x.prev_ax * h / w
                - x.prev_ay / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P.get("Cl2") * x.vx * x.vx;

    double N_rr = 0.5 * mr * g + 0.5 * m * x.prev_ax * h / w
                + x.prev_ay / t_rear * ( mr * h_roll_r + Kr / K_total * (mf * h_prim_f + mr * h_prim_r))
                + 0.5 * P.get("Cl2") * x.vx * x.vx;

    const double FZ_MIN = 50.0;
    N_fl = std::max(N_fl, FZ_MIN);
    N_fr = std::max(N_fr, FZ_MIN);
    N_rl = std::max(N_rl, FZ_MIN);
    N_rr = std::max(N_rr, FZ_MIN);

    // ============================================================
    // 7) Kąty poślizgu
    // ============================================================
    double alpha_fr = -std::atan2(vy_fr, safe_vx_signed(vx_fr));
    double alpha_fl = -std::atan2(vy_fl, safe_vx_signed(vx_fl));
    double alpha_rr = -std::atan2(vy_rr, safe_vx_signed(vx_rr));
    double alpha_rl = -std::atan2(vy_rl, safe_vx_signed(vx_rl));

    alpha_fr = std::clamp(alpha_fr, -M_PI/2 + 0.3, M_PI/2 - 0.3);
    alpha_fl = std::clamp(alpha_fl, -M_PI/2 + 0.3, M_PI/2 - 0.3);
    alpha_rr = std::clamp(alpha_rr, -M_PI/2 + 0.3, M_PI/2 - 0.3);
    alpha_rl = std::clamp(alpha_rl, -M_PI/2 + 0.3, M_PI/2 - 0.3);

    // ============================================================
    // 8) Slip ratio + deadband
    // ============================================================
    const double KAPPA_DEADBAND = 0.005;

    auto apply_kappa_deadband = [&](double k) -> double {
        return (std::abs(k) < KAPPA_DEADBAND) ? 0.0 : k;
    };

    const double kappa_rr_raw = std::clamp((x.omega_rr * R - vx_rr) / vx_rr_den_kappa, -0.99, 0.99);
    const double kappa_rl_raw = std::clamp((x.omega_rl * R - vx_rl) / vx_rl_den_kappa, -0.99, 0.99);
    const double kappa_fr_raw = std::clamp((x.omega_fr * R - vx_fr) / vx_fr_den_kappa, -0.99, 0.99);
    const double kappa_fl_raw = std::clamp((x.omega_fl * R - vx_fl) / vx_fl_den_kappa, -0.99, 0.99);

    const double kappa_rr = apply_kappa_deadband(kappa_rr_raw);
    const double kappa_rl = apply_kappa_deadband(kappa_rl_raw);
    const double kappa_fr = apply_kappa_deadband(kappa_fr_raw);
    const double kappa_fl = apply_kappa_deadband(kappa_fl_raw);

    // ============================================================
    // 9) dfz
    // ============================================================
    const double dfz_fl = (N_fl - N0) / N0;
    const double dfz_fr = (N_fr - N0) / N0;
    const double dfz_rl = (N_rl - N0) / N0;
    const double dfz_rr = (N_rr - N0) / N0;

    // ============================================================
    // 10) Helper: MF6.1 pure + pochodna dF/dx
    // ============================================================
    auto mf61_pure = [&](double x_in, double C, double D, double B, double E, double Sv,
                         double& F, double& dFdx)
    {
        const double Bx = B * x_in;
        const double atanBx = std::atan(Bx);
        const double phi = Bx - E * (Bx - atanBx);
        const double atanPhi = std::atan(phi);

        F = D * std::sin(C * atanPhi) + Sv;

        const double d_atanPhi_dphi = 1.0 / (1.0 + phi * phi);
        const double d_atanBx_dBx   = 1.0 / (1.0 + Bx * Bx);

        const double dphi_dx = B - E * (B - B * d_atanBx_dBx);

        dFdx = D * std::cos(C * atanPhi) * C * d_atanPhi_dphi * dphi_dx;
    };

    // ============================================================
    // 11) Helper: MF6.1 pure + coupling elipsą
    // ============================================================
    auto compute_wheel_forces = [&](double Fz, double dfz,
                                    double kappa, double alpha,
                                    bool allow_longitudinal,
                                    double& Fx, double& Fy,
                                    double& dFx_dkappa, double& dFy_dalpha,
                                    double& mu_x, double& mu_y)
    {
        // ---------- LONG pure ----------
        double Cx, mux, Dx, Kxk, Bx, Ex, Shx, Svx;

        {
            Cx = pCx1;

            mux = lambda_x * (pDx1 + pDx2 * dfz);
            mux = std::max(mux, 1e-6);
            Dx  = mux * Fz;

            Kxk = Fz * (pKx1 + pKx2 * dfz) * std::exp(pKx3 * dfz);

            Shx = pHx1 + pHx2 * dfz;
            Svx = Fz * (pVx1 + pVx2 * dfz);

            Svx = 0.0;

            const double denom = std::max(Cx * Dx, 1e-9);
            Bx = Kxk / denom;

            const double E0 = (pEx1 + pEx2 * dfz + pEx3 * dfz * dfz);
            auto smooth_sign = [&](double xx){
                const double s = 50.0;
                return std::tanh(s * xx);
            };
            Ex = E0 * (1.0 - pEx4 * smooth_sign(kappa + Shx));
        }

        double Fx0 = 0.0;
        double dFx0_dk = 0.0;

        if (allow_longitudinal)
        {
            const double kx = kappa + Shx;
            mf61_pure(kx, Cx, Dx, Bx, Ex, Svx, Fx0, dFx0_dk);

            double Fx_raw = Fx0;
            double Fx_bias = 0.0, dtmp = 0.0;
            {
                const double kx0 = 0.0 + Shx;
                mf61_pure(kx0, Cx, Dx, Bx, Ex, Svx, Fx_bias, dtmp);
            }

            Fx0 = Fx_raw - Fx_bias;
        }

        // ---------- LAT pure ----------
        double Cy, muy, Dy, Kya, By, Ey, Shy, Svy;

        {
            Cy = pCy1;

            muy = lambda_y * (pDy1 + pDy2 * dfz);
            muy = std::max(muy, 1e-6);
            Dy  = muy * Fz;

            Kya = pKy1 * N0 * std::sin(pKy4 * std::atan(Fz / (pKy2 * N0)));

            const double denom = std::max(Cy * Dy, 1e-9);
            By = Kya / denom;

            Shy = pHy1 + pHy2 * dfz;
            Svy = Fz * (pVy1 + pVy2 * dfz);

            Svy = 0.0;

            auto smooth_sign = [&](double xx){
                const double s = 50.0;
                return std::tanh(s * xx);
            };
            Ey = (pEy1 + pEy2 * dfz) * (1.0 - pEy3 * smooth_sign(alpha + Shy));
        }

        const double ay = alpha + Shy;
        double Fy0 = 0.0;
        double dFy0_da = 0.0;
        mf61_pure(ay, Cy, Dy, By, Ey, Svy, Fy0, dFy0_da);

        double Fy_raw = Fy0;
        double Fy_bias = 0.0, dtmp2 = 0.0;
        {
            const double ay0 = 0.0 + Shy;
            mf61_pure(ay0, Cy, Dy, By, Ey, Svy, Fy_bias, dtmp2);
        }
        Fy0 = Fy_raw - Fy_bias;

        // ---------- COUPLING ELLIPSE ----------
        const double FxMax = mux * Fz;
        const double FyMax = muy * Fz;

        const double eps = 1e-9;
        const double FxMaxSafe = std::max(std::abs(FxMax), eps);
        const double FyMaxSafe = std::max(std::abs(FyMax), eps);

        const double axu = Fx0 / FxMaxSafe;
        const double ayu = Fy0 / FyMaxSafe;

        const double s = std::sqrt(axu * axu + ayu * ayu);

        if (s <= 1.0 || !std::isfinite(s))
        {
            Fx = Fx0;
            Fy = Fy0;
        }
        else
        {
            const double invs = 1.0 / s;
            Fx = Fx0 * invs;
            Fy = Fy0 * invs;
        }

        // pochodne do relaksacji biorę z pure MF
        dFx_dkappa = dFx0_dk;
        dFy_dalpha = dFy0_da;

        mu_x = mux;
        mu_y = muy;
    };

    // ============================================================
    // 12) Model Coulombowski
    // ============================================================
    auto compute_coulomb_forces = [&](double Fz, double dfz, double vx_loc, double vy_loc, double omega,
                                      double& Fxc, double& Fyc)
    {
        const double vsx = omega * R - vx_loc;
        const double vsy = -vy_loc;

        const double mu_x = lambda_x * (pDx1 + pDx2 * dfz);
        const double mu_y = lambda_y * (pDy1 + pDy2 * dfz);

        const double Fx_max = std::max(mu_x * Fz, 1e-3);
        const double Fy_max = std::max(mu_y * Fz, 1e-3);

        const double K_stiff = 800.0;

        Fxc = K_stiff * vsx;
        Fyc = K_stiff * vsy;

        Fxc = std::clamp(Fxc, -Fx_max, Fx_max);
        Fyc = std::clamp(Fyc, -Fy_max, Fy_max);
    };

    // Zmienne dla Coulomba
    double Fxc_fl = 0.0, Fyc_fl = 0.0, Fxc_fr = 0.0, Fyc_fr = 0.0;
    double Fxc_rl = 0.0, Fyc_rl = 0.0, Fxc_rr = 0.0, Fyc_rr = 0.0;

    // Zmienne dla MF6.1
    double Fx_fl = 0.0, Fy_fl = 0.0, dFx_fl_dk = 0.0, dFy_fl_da = 0.0, mux_fl = 0.0, muy_fl = 0.0;
    double Fx_fr = 0.0, Fy_fr = 0.0, dFx_fr_dk = 0.0, dFy_fr_da = 0.0, mux_fr = 0.0, muy_fr = 0.0;
    double Fx_rl = 0.0, Fy_rl = 0.0, dFx_rl_dk = 0.0, dFy_rl_da = 0.0, mux_rl = 0.0, muy_rl = 0.0;
    double Fx_rr = 0.0, Fy_rr = 0.0, dFx_rr_dk = 0.0, dFy_rr_da = 0.0, mux_rr = 0.0, muy_rr = 0.0;

    // ============================================================
    // 13) Obliczanie sił z obu modeli
    // ============================================================
    compute_coulomb_forces(N_fl, dfz_fl, vx_fl, vy_fl, x.omega_fl, Fxc_fl, Fyc_fl);
    compute_coulomb_forces(N_fr, dfz_fr, vx_fr, vy_fr, x.omega_fr, Fxc_fr, Fyc_fr);
    compute_coulomb_forces(N_rl, dfz_rl, vx_rl, vy_rl, x.omega_rl, Fxc_rl, Fyc_rl);
    compute_coulomb_forces(N_rr, dfz_rr, vx_rr, vy_rr, x.omega_rr, Fxc_rr, Fyc_rr);

    compute_wheel_forces(
        N_fl, dfz_fl, kappa_fl, alpha_fl,
        std::abs(kappa_fl_raw) >= KAPPA_DEADBAND,
        Fx_fl, Fy_fl, dFx_fl_dk, dFy_fl_da, mux_fl, muy_fl
    );

    compute_wheel_forces(
        N_fr, dfz_fr, kappa_fr, alpha_fr,
        std::abs(kappa_fr_raw) >= KAPPA_DEADBAND,
        Fx_fr, Fy_fr, dFx_fr_dk, dFy_fr_da, mux_fr, muy_fr
    );

    compute_wheel_forces(
        N_rl, dfz_rl, kappa_rl, alpha_rl,
        std::abs(kappa_rl_raw) >= KAPPA_DEADBAND,
        Fx_rl, Fy_rl, dFx_rl_dk, dFy_rl_da, mux_rl, muy_rl
    );

    compute_wheel_forces(
        N_rr, dfz_rr, kappa_rr, alpha_rr,
        std::abs(kappa_rr_raw) >= KAPPA_DEADBAND,
        Fx_rr, Fy_rr, dFx_rr_dk, dFy_rr_da, mux_rr, muy_rr
    );

    // ============================================================
    // 14) BLENDING - obliczanie wag przejścia
    // ============================================================
    const double V_TRANS_LOW  = 3.0;
    const double V_TRANS_HIGH = 4.0;

    double abs_vx = std::abs(x.vx);
    double w_mf = 0.0;

    if (abs_vx >= V_TRANS_HIGH) {
        w_mf = 1.0;
    } else if (abs_vx > V_TRANS_LOW) {
        double t = (abs_vx - V_TRANS_LOW) / (V_TRANS_HIGH - V_TRANS_LOW);
        w_mf = t * t * (3.0 - 2.0 * t);
    }
    double w_c = 1.0 - w_mf;

    // ============================================================
    // 15) Obliczanie długości relaksacji
    // ============================================================
    const double Vx_rr = speed_mag(vx_rr);
    const double Vx_rl = speed_mag(vx_rl);
    const double Vx_fl = speed_mag(vx_fl);
    const double Vx_fr = speed_mag(vx_fr);

    double dyn_L_alpha_fl = Vx_fl * dy / cy + (1.0 / cy) * std::abs(dFy_fl_da);
    double dyn_L_alpha_fr = Vx_fr * dy / cy + (1.0 / cy) * std::abs(dFy_fr_da);
    double dyn_L_alpha_rl = Vx_rl * dy / cy + (1.0 / cy) * std::abs(dFy_rl_da);
    double dyn_L_alpha_rr = Vx_rr * dy / cy + (1.0 / cy) * std::abs(dFy_rr_da);

    double dyn_L_kappa_fl = Vx_fl * dx / cx + (1.0 / cx) * std::abs(dFx_fl_dk);
    double dyn_L_kappa_fr = Vx_fr * dx / cx + (1.0 / cx) * std::abs(dFx_fr_dk);
    double dyn_L_kappa_rl = Vx_rl * dx / cx + (1.0 / cx) * std::abs(dFx_rl_dk);
    double dyn_L_kappa_rr = Vx_rr * dx / cx + (1.0 / cx) * std::abs(dFx_rr_dk);

    // ============================================================
    // 16) Połączona integracja
    // ============================================================
    const double inv_dt = (dt > 1e-6) ? (1.0 / dt) : 0.0;

    auto calc_blended_deriv = [&](double F_state,
                                  double Fc,
                                  double Fmf,
                                  double vx_loc,
                                  double L) -> double
    {
        const double dF_coulomb = (Fc - F_state) * inv_dt;

        const double V  = std::max(std::abs(vx_loc), 0.1);
        const double Ls = std::max(L, L_RELAX_MIN);

        const double z = std::min((V / Ls) * dt, Z_MAX);
        const double one_minus_a = -std::expm1(-z);

        const double dF_mf = (Fmf - F_state) * (one_minus_a * inv_dt);

        return (w_c * dF_coulomb) + (w_mf * dF_mf);
    };

    temp.fy_fl = calc_blended_deriv(x.fy_fl, Fyc_fl, Fy_fl, vx_fl, dyn_L_alpha_fl);
    temp.fy_fr = calc_blended_deriv(x.fy_fr, Fyc_fr, Fy_fr, vx_fr, dyn_L_alpha_fr);
    temp.fy_rl = calc_blended_deriv(x.fy_rl, Fyc_rl, Fy_rl, vx_rl, dyn_L_alpha_rl);
    temp.fy_rr = calc_blended_deriv(x.fy_rr, Fyc_rr, Fy_rr, vx_rr, dyn_L_alpha_rr);

    temp.fx_fl = calc_blended_deriv(x.fx_fl, Fxc_fl, Fx_fl, vx_fl, dyn_L_kappa_fl);
    temp.fx_fr = calc_blended_deriv(x.fx_fr, Fxc_fr, Fx_fr, vx_fr, dyn_L_kappa_fr);
    temp.fx_rl = calc_blended_deriv(x.fx_rl, Fxc_rl, Fx_rl, vx_rl, dyn_L_kappa_rl);
    temp.fx_rr = calc_blended_deriv(x.fx_rr, Fxc_rr, Fx_rr, vx_rr, dyn_L_kappa_rr);

    return temp;
}

} // namespace lem_dynamics_sim_