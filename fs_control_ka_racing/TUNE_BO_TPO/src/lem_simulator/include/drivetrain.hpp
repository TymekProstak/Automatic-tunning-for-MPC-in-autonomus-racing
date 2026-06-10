#pragma once
#include "ParamBank.hpp"
#include "uttilities.hpp"

// zakładamy 1 rzędowy model drivetrainu
namespace lem_dynamics_sim_{


    inline State derative_drivetrain(const ParamBank& P, const  State& x, const Input& u) {
        State temp;
        temp.setZero();
        // first order system per wheel (4x4)
        const double tau = P.get("drivetrain_timescale");
        temp.torque_rr = (u.torque_request_rr - x.torque_rr) / tau;
        temp.torque_rl = (u.torque_request_rl - x.torque_rl) / tau;
        temp.torque_fr = (u.torque_request_fr - x.torque_fr) / tau;
        temp.torque_fl = (u.torque_request_fl - x.torque_fl) / tau;

        return temp;
    }


    
}
