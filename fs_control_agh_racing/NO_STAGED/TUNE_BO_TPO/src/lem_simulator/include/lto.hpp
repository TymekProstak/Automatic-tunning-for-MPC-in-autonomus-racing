#pragma once

#include <vector>
#include <Eigen/Dense>

// MUSI być przed użyciem v2_control::
#include "ParamBank_lto.hpp"
#include "spline.hpp"
#include "Vec2.hpp"
namespace lto {

using ParamBank = lto::ParamBank_lto;


// ============================================================
//  Parametry LTO
// ============================================================
struct LtoParams
{
    double g       = 9.81;

    double v_min   = 1.0;
    double v_max   = 40.0;

    double lf      = 1.5;
    double lr      = 1.5;

    // Napęd / opory
    double Cm      = 1.0;
    double Cr0     = 0.1;
    double Cl      = 1.0;
    double Cd      = 1.0;

    double max_drive_power = 80000.0;
    double max_brake_power = 80000.0;

    double max_delta   = 0.4;
    double min_delta   = -0.4;

    double max_d_delta = 0.8;
    double min_d_delta = -0.8;

    double max_d_T     = 500.0;
    double min_d_T     = -500.0;
    
    double max_tv      = 2000.0;
    double min_tv      = -2000.0;

    double Fz_nom      = 1000.0;

    // Limity opon / przyczepność
    double mu_y = 1.6;
    double mu_x = 1.6;

    // “MF-like”
    double C = 1.5;
    double D = 1.65; 
    double B = 18.6;

    // Wymiary auta do constraintów toru
    double length = 3.0;
    double width = 1.5;     
    double track_width = 1.2;

    // Koszty
    double d_delta_cost = 600.0;
    double d_T_cost     = 1.0;
    double beta_cost    = 1.0;
    double s_dot_cost   = 1.0;
    double tv_cost      = 1.0;

    // Regularizacja
    double w_ax = 1e-3;

    double m  = 193.0;
    double Iz = 85.0;

    // LTO spatial step [m]
    double ds = 0.5;

    // tylko do initial guess dt (żeby guess nie eksplodował)
    double s_dot_guess_floor = 2.0;

    // *** TWARDY constraint w NLP ***
    double s_dot_min = 2.0;   // [m/s]

    double saftey_factor = 0.95; // mnożnik na oponę, żeby mieć margines bezpieczeństwa (np. 0.95 to 95% przyczepności)

    void load_lto_param_from_param_bank(const ParamBank& P);
};

// ============================================================
//  Stany / sterowania LTO
// ============================================================
struct MPC_State_LTO
{
    double ey       = 0.0;
    double epsi     = 0.0;
    double vx       = 0.0;
    double vy       = 0.0;
    double r        = 0.0;
    double delta    = 0.0;
    double throthle = 0.0;

    double s     = 0.0;   // s siatki (k*ds) użyte w NLP
    double kappa = 0.0;   // kappa(ref) użyte w NLP (do obliczeń)
    double s_dot = 0.0;   // surowe s_dot
};

struct MPC_Action_LTO
{
    double ddelta_opt = 0.0;
    double dthot_opt  = 0.0; // u Ciebie to d_T
    double tv_opt     = 0.0;
};

// ============================================================
//  Wynik: ZWRACAM GEOMETRIĘ ZOPTYMALIZOWANEJ ŚCIEŻKI (world-frame)
// ============================================================
struct LtoResult
{
    // NLP wynik
    std::vector<MPC_State_LTO>  states;    // size = N+1
    std::vector<MPC_Action_LTO> actions;   // size = N
    std::vector<double>         vx_list;   // size = N (bez duplikatu końca)

    // *** ZOPTYMALIZOWANA ŚCIEŻKA (bez duplikatu końca) ***
    // Rozmiar = N (jedno okrążenie)
    Eigen::VectorXd s_opt;      // [m] kumulatywna długość po chordach na opt ścieżce, w [0, L_opt)
    Eigen::VectorXd x_opt;      // [m]
    Eigen::VectorXd y_opt;      // [m]
    Eigen::VectorXd yaw_opt;    // [rad]
    Eigen::VectorXd kappa_opt;  // [1/m] (z dyskretnej geometrii opt ścieżki)
    double total_length_opt = 0.0;
    Eigen::VectorXd lateral_deviation; // [m] (odległość od ref w normal_left(ref))
    // Status solvera
    int  ipopt_status  = 0;
    bool ipopt_success = false;
};

// ============================================================
//  Publiczny interfejs
// ============================================================

LtoResult solve_lto_speed_profile(
    const Eigen::VectorXd& X_path,
    const Eigen::VectorXd& Y_path,
    const ParamBank& P);

LtoResult solve_lto_speed_profile(
    const Eigen::VectorXd& X_path,
    const Eigen::VectorXd& Y_path,
    const LtoParams& prm);

} // namespace lto
