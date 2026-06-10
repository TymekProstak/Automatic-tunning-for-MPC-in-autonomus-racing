#include "steering_system.hpp"
 namespace lem_dynamics_sim_
{

    State  derative_steering(const ParamBank& P, const State& x, const Input& u) {
        State temp;
        temp.setZero();

        double omega_n = P.get("natural_frequency_steering_system")  ;
        double omega_n_pow = omega_n*omega_n;

        temp.rack_angle = x.d_rack_angle ;
        temp.d_rack_angle = -2*P.get("steering_system_damping")*omega_n*x.d_rack_angle - omega_n_pow*x.rack_angle + u.rack_angle_request*omega_n_pow ;

       // Pobieramy chwilowe przełożenie geometryczne z modelu CAD: d(delta)/d(rack)
       delta_angels steering_ratio = derative_with_respect_to_rack_angle_anti_akerman(x.rack_angle, P);

       temp.delta_left  = steering_ratio.delta_left  * x.d_rack_angle;
       temp.delta_right = steering_ratio.delta_right * x.d_rack_angle;
       

       temp.d_delta_left  = steering_ratio.delta_left  * temp.d_rack_angle;
       temp.d_delta_right = steering_ratio.delta_right * temp.d_rack_angle;


        
        return temp;
    }
    
} // namespace namespace lem_dynamics_sim_
