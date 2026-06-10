#pragma once

#include "utilities.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

#include "ParamBank.hpp"

namespace v2_control
{
    // ============================================================
    // Zamiast stałych z ACADOS:
    // - NX: stałe (model ma 7 stanów)
    // - NU: w tej wersji UNCONSTRAINED liczę tylko u = d(delta_request)/dt
    // - N: horizon biorę runtime z ParamBank (N_horizon_)
    // ============================================================
    constexpr int NX = 7;
    constexpr int NU = 1;


    // ============================================================
    // Struktura stanu MPC (7 stanów jak wcześniej)
    // ============================================================
    struct MPC_State
    {
        double ey;
        double epsi;
        double vy;
        double r;
        double delta;
        double d_delta;
        double delta_request;

        double vx0_body = 0.0; // extra

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
        double mtv_opt;          // w tej wersji: 0.0 (nie liczę torque vectoringu)
        bool   success;
        double next_yaw_rate;
    };

    class MPCInterface
    {
    public:
        MPCInterface();
        MPCInterface(const ParamBank &P);
        ~MPCInterface();

        MPC_Return solve(const MPC_State &x0,
                         const Eigen::VectorXd &curvature_ref,
                         const Eigen::VectorXd &velocity_ref);

        MPC_Return solve(const MPC_State &x0,
                         const Eigen::VectorXd &curvature_ref,
                         const Eigen::VectorXd &velocity_ref,
                         const Eigen::VectorXd &acceleration_ref,
                         double vx0_body);

    private:
        // (zostawiam dla kompatybilności semantycznej – u mnie to tylko reset cache)
        void reset_initial_guess(const MPC_State& x0,
                                 const std::vector<double>& vref_vec);

        // LTI continuous model (jak u Ciebie)
        void build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>& Ac,
                                           Eigen::Matrix<double, NX, NU>& Bc,
                                           double v_ref0) const;

        void build_lti_continuous_matrices(Eigen::Matrix<double, NX, NX>& Ac,
                                           Eigen::Matrix<double, NX, NU>& Bc) const;

    private:
        ParamBank param_;
        bool is_initialized_ = false;

        int    N_ = 60;     // runtime horizon

        // (opcjonalnie) trzymam ostatnią trajektorię u do debug/warmstartu
        std::vector<Eigen::Matrix<double, NU, 1>> last_output;
    };

} // namespace v2_control