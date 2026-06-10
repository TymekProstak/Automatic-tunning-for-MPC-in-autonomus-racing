#pragma once

#include "utilities.hpp"
#include <Eigen/Dense>
#include <vector>
#include "ParamBank.hpp"
#include "spline.hpp"   // <- MUSI być, bo używasz TrackSpline2D

extern "C" {
    #include "acados_solver_mpc_ltv_discrete.h"
    #include "acados_c/ocp_nlp_interface.h"
}

namespace v2_control
{
    // Stałe zdefiniowane w wygenerowanym kodzie Acadosa
    constexpr int NX = MPC_LTV_DISCRETE_NX;
    constexpr int NU = MPC_LTV_DISCRETE_NU;
    constexpr int N  = MPC_LTV_DISCRETE_N;
    constexpr int NP = MPC_LTV_DISCRETE_NP;

    // ============================================================
    // Struktura stanu MPC (MUSI pasować do modelu Acados: 7 stanów)
    // Dodatkowe pola (vx0_body) są OK, ale NIE idą do Acadosa.
    // ============================================================
    struct MPC_State
    {
        // --- stany Acados (NX = 7) ---
        double ey;
        double epsi;
        double vy;
        double r;
        double delta;
        double d_delta;
        double delta_request;

        // --- extra (nie jest stanem Acados) ---
        double vx0_body = 0.0;

        void to_array(double* data) const {
            data[0] = ey;
            data[1] = epsi;
            data[2] = vy;
            data[3] = r;
            data[4] = delta;
            data[5] = d_delta;
            data[6] = delta_request;
        }

        void to_eigen(Eigen::Matrix<double, NX, 1>& vec) const {
            vec(0) = ey;
            vec(1) = epsi;
            vec(2) = vy;
            vec(3) = r;
            vec(4) = delta;
            vec(5) = d_delta;
            vec(6) = delta_request;
        }
    };

    struct MPC_Return
    {
        double ddelta_opt;
        double mtv_opt;
        bool   success;
        double next_yaw_rate;
    };

    class MPCInterface
    {
    public:
        MPCInterface();
        MPCInterface(const ParamBank &P);
        ~MPCInterface();

        // ============================================================
        // VARIANT-B (Frenet s_dot) + kappa z TrackSpline2D
        //
        // - spline_ dostajesz przez set_Spline()
        // - solve dostaje s0 (np. z projekcji / trackingu)
        // - velocity_ref: używane tylko do a_long fallback (finite diff) i vx fallback
        // - acceleration_ref: a_long_ref (jeśli dostępne)
        // - vx0_body: preferuj odometrię (albo x0.vx0_body), fallback -> velocity_ref(0)
        // ============================================================

        MPC_Return solve(const MPC_State &x0,
            const TrackSpline2D& track,
            double s0,
            const Eigen::VectorXd &velocity_ref);

        MPC_Return solve(const MPC_State &x0,
            const TrackSpline2D& track,
            double s0,
            const Eigen::VectorXd &velocity_ref,
            const Eigen::VectorXd &acceleration_ref,
            double vx_body);

        void set_Spline(const TrackSpline2D& spline) { spline_ = spline; }

    private:
        // ============================================================
        // Helpers
        // ============================================================
        void reset_initial_guess_splitv_(const MPC_State& x0,
                                         const std::vector<double>& v_vehicle_pred);

        void set_cost_to_acados();

        // ============================================================
        // LTV (VARIANT-B): kappa_k = spline_.getCurvature(s_k), gdzie s_k roll-out z s_dot
        // ============================================================
        void ltv_matrixes_to_acados_splitv_(
            const MPC_State& x0,
            const TrackSpline2D& track,
            double s0,
            const std::vector<double>& v_vehicle_vec);
        // ============================================================
        // Jacobian (VARIANT-B)
        // 2nd arg = kappa (NIE v_path)
        // ============================================================
        void calculate_continuous_jacobian_splitv_(
            const Eigen::Matrix<double, NX, 1>& x,
            double kappa,
            double v_vehicle,
            Eigen::Matrix<double, NX, NX>& Ac,
            Eigen::Matrix<double, NX, NU>& Bc);

        // ============================================================
        // (opcjonalne) debug/stubs — możesz zostawić jeśli masz definicje,
        // albo usunąć, jeśli już nieużywane.
        // ============================================================
        void print_problem_debug(const Eigen::Matrix<double, NX, NX> &Ad,
                                 const Eigen::Matrix<double, NX, NU> &Bd,
                                 const Eigen::Matrix<double, NX, 1>  &Kd,
                                 const MPC_State &x0);

        void build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>& Ac,
                                           Eigen::Matrix<double, NX, NU>& Bc) const;

        void build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>& Ac,
                                           Eigen::Matrix<double, NX, NU>& Bc,
                                           double v_ref0) const;

        void push_lti_params_to_acados(double v_ref0);

    private:
        ParamBank param_;
        bool is_initialized_ = false;

        std::vector<Eigen::Matrix<double, NU, 1>> last_output;
        std::vector<double> lti_p_vec_;

        // ACADOS handles
        mpc_ltv_discrete_solver_capsule *capsule_ = nullptr;
        ocp_nlp_config *nlp_config_ = nullptr;
        ocp_nlp_dims   *nlp_dims_   = nullptr;
        ocp_nlp_in     *nlp_in_     = nullptr;
        ocp_nlp_out    *nlp_out_    = nullptr;
        ocp_nlp_solver *nlp_solver_ = nullptr;

        // Track spline (periodic / closed — zakładasz closed)
        TrackSpline2D spline_;
    };

} // namespace v2_control