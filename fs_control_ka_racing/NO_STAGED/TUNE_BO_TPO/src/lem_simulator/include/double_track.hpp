#pragma once
#include "drivetrain.hpp"
#include "tire_model.hpp"
#include "steering_system.hpp"
#include "uttilities.hpp"
#include "wheels_dynamis_model.hpp"
#include <algorithm> 


namespace lem_dynamics_sim_{

    struct Log_Info_reduced{

        double vx; // m/s
        double vy; // m/s
        double yaw_rate; // rad/s
        double x; // m
        double y; // m
        double yaw; // rad

        double rack_angle; // deg

        // 4x4 slip ratios
        double kappa_fl; // nondim
        double kappa_fr; // nondim
        double kappa_rl; // nondim
        double kappa_rr; // nondim

        double time; // s
        double rack_angle_request; // deg

        // 4 torque requests
        double torque_request_fl; // Nm
        double torque_request_fr; // Nm
        double torque_request_rl; // Nm
        double torque_request_rr; // Nm

        double ax; // g units
        double ay; // g units
        
    };

    struct Log_Info_full{


        double kappa_fl; // nondim
        double kappa_fr; // nondim
        double kappa_rl; // nondim
        double kappa_rr; // nondim

        double slip_angle_fl; // deg
        double slip_angle_fr; // deg
        double slip_angle_rl; // deg
        double slip_angle_rr; // deg

        double slip_angle_body; // deg

        double fz_fl; // N
        double fz_fr; // N
        double fz_rl; // N
        double fz_rr; // N

        double fy_fl; // N
        double fy_fr; // N
        double fy_rl; // N
        double fy_rr; // N

        double fx_fl; // N
        double fx_fr; // N
        double fx_rl; // N
        double fx_rr; // N

        // per-wheel torques
        double torque_fl; // Nm
        double torque_fr; // Nm
        double torque_rl; // Nm
        double torque_rr; // Nm

        // wheel speeds
        double omega_fl; // rad/s
        double omega_fr; // rad/s
        double omega_rl; // rad/s
        double omega_rr; // rad/s

        double delta_left; // deg
        double delta_rigth; // deg
        double rack_angle; // deg

        double ax; // g units
        double ay; // g units

        double yaw_rate; // rad/s
        double vx; // m/s
        double vy; // m/s
        double x; // m
        double y; // m
        double yaw; // rad

        double Power_total; // kW

        double rack_angle_request; // deg

        // per-wheel torque requests
        double torque_request_fl; // Nm
        double torque_request_fr; // Nm
        double torque_request_rl; // Nm
        double torque_request_rr; // Nm

        double time; // s

        double total_drag; // N
        double total_downforce; // N
        

    };

    Log_Info_reduced log_info_reduced(const State& x, const Input& u, const ParamBank& P , int step_number);

    State model_derative(const ParamBank& P, const  State& x, const Input& u);
    Log_Info_full log_info_full(const State& x, const Input& u, const ParamBank& P , int step_number);
    void rk4_sim_timestep(State& x, const Input& u, const ParamBank& P);
    void euler_sim_timestep(State& x, const Input& u, const ParamBank& P);


} // namesapce lem_dynamis_sim_
