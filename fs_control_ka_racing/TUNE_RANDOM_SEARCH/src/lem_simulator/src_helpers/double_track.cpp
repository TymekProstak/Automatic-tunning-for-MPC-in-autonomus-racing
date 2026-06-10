#include "double_track.hpp"

namespace lem_dynamics_sim_{

    namespace {

        inline double clamp01_(double x)
        {
            return std::max(0.0, std::min(1.0, x));
        }
        
        inline double smoothstep01_(double x)
        {
            x = clamp01_(x);
            return x * x * (3.0 - 2.0 * x);
        }
        
        // w_direct = 1  -> pełne Fx = T/R
        // w_direct = 0  -> pełne Fx z modelu opony
        inline double low_torque_direct_weight_(double abs_torque_nm,
                                                double torque_on_nm,
                                                double torque_off_nm)
        {
            if (torque_off_nm <= torque_on_nm) {
                return (abs_torque_nm <= torque_on_nm) ? 1.0 : 0.0;
            }
        
            if (abs_torque_nm <= torque_on_nm) return 1.0;
            if (abs_torque_nm >= torque_off_nm) return 0.0;
        
            const double xi = (abs_torque_nm - torque_on_nm) / (torque_off_nm - torque_on_nm);
            return 1.0 - smoothstep01_(xi);
        }
        
        inline double effective_fx_from_torque_blend_(double fx_tire,
                                                      double actual_torque_nm,
                                                      double wheel_radius_m,
                                                      double torque_on_nm,
                                                      double torque_off_nm)
        {
            const double w_direct = low_torque_direct_weight_(std::abs(actual_torque_nm),
                                                              torque_on_nm,
                                                              torque_off_nm);
        
            const double fx_direct = actual_torque_nm / std::max(wheel_radius_m, 1e-9);
            return w_direct * fx_direct + (1.0 - w_direct) * fx_tire;
        }
        
        } // namespace

        State model_derative(const ParamBank& P, const State& x, const Input& u)
        {
            const double R = P.get("R");
        
            // low-torque override:
            // poniżej torque_on -> praktycznie czyste Fx = T/R
            // powyżej torque_off -> praktycznie czyste Fx z modelu opony
            const double torque_direct_on_nm  = 3.0;
            const double torque_direct_off_nm = 6.0;
        
            // =========================================================
            // 1) Efektywne siły wzdłużne na kołach
            // =========================================================
            const double fx_fl_eff = effective_fx_from_torque_blend_(
                x.fx_fl, x.torque_fl, R, torque_direct_on_nm, torque_direct_off_nm);
            const double fx_fr_eff = effective_fx_from_torque_blend_(
                x.fx_fr, x.torque_fr, R, torque_direct_on_nm, torque_direct_off_nm);
            const double fx_rl_eff = effective_fx_from_torque_blend_(
                x.fx_rl, x.torque_rl, R, torque_direct_on_nm, torque_direct_off_nm);
            const double fx_rr_eff = effective_fx_from_torque_blend_(
                x.fx_rr, x.torque_rr, R, torque_direct_on_nm, torque_direct_off_nm);
        
            // =========================================================
            // 2) Aero / opory
            // =========================================================
            const double areo_drag             = P.get("Cd") * x.vx * x.vx;
            const double areo_downforce_front  = P.get("Cl1") * x.vx * x.vx;
            const double areo_downforce_rear   = P.get("Cl2") * x.vx * x.vx;
            const double areo_downforce        = areo_downforce_front + areo_downforce_rear;
            const double rolling_resistance    = P.get("Cr") * (P.get("m") * P.get("g") + areo_downforce);
        
            // =========================================================
            // 3) Transformacja sił koła -> body
            //    UWAGA: podmieniam tylko Fx, Fy zostawiam z modelu opony
            // =========================================================
            const double Fx_fl_b = fx_fl_eff * std::cos(x.delta_left)  - x.fy_fl * std::sin(x.delta_left);
            const double Fy_fl_b = fx_fl_eff * std::sin(x.delta_left)  + x.fy_fl * std::cos(x.delta_left);
        
            const double Fx_fr_b = fx_fr_eff * std::cos(x.delta_right) - x.fy_fr * std::sin(x.delta_right);
            const double Fy_fr_b = fx_fr_eff * std::sin(x.delta_right) + x.fy_fr * std::cos(x.delta_right);
        
            const double Fx_rl_b = fx_rl_eff;
            const double Fy_rl_b = x.fy_rl;
        
            const double Fx_rr_b = fx_rr_eff;
            const double Fy_rr_b = x.fy_rr;
        
            const double sgn_vx   = (x.vx == 0.0) ? 0.0 : std::copysign(1.0, x.vx);
            const double F_resist = (areo_drag + rolling_resistance) * sgn_vx;
        
            const double Fx_total = Fx_fl_b + Fx_fr_b + Fx_rl_b + Fx_rr_b - F_resist;
            const double Fy_total = Fy_fl_b + Fy_fr_b + Fy_rl_b + Fy_rr_b;
        
            // =========================================================
            // 4) Moment yaw od sił w body
            // =========================================================
            const double a   = P.get("a");
            const double b   = P.get("b");
            const double tf2 = 0.5 * P.get("t_front");
            const double tr2 = 0.5 * P.get("t_rear");
        
            double Mz = 0.0;
            Mz += a    * Fy_fl_b - (+tf2) * Fx_fl_b; // FL
            Mz += a    * Fy_fr_b - (-tf2) * Fx_fr_b; // FR
            Mz += (-b) * Fy_rl_b - (+tr2) * Fx_rl_b; // RL
            Mz += (-b) * Fy_rr_b - (-tr2) * Fx_rr_b; // RR
        
            // =========================================================
            // 5) Składanie pochodnej
            // =========================================================
            State temp;
            temp.setZero();
        
            temp.x        = x.vx * std::cos(x.yaw) - x.vy * std::sin(x.yaw);
            temp.y        = x.vy * std::cos(x.yaw) + x.vx * std::sin(x.yaw);
            temp.yaw      = x.yaw_rate;
            temp.vx       = Fx_total / P.get("m") + x.vy * x.yaw_rate;
            temp.vy       = Fy_total / P.get("m") - x.vx * x.yaw_rate;
            temp.yaw_rate = Mz / P.get("Iz");
        
            // zostawiam jak miałeś
            temp.prev_ax = (Fx_total / P.get("m") - x.prev_ax) / P.get("simulation_time_step");
            temp.prev_ay = (Fy_total / P.get("m") - x.prev_ay) / P.get("simulation_time_step");
        
            temp += derative_steering(P, x, u);
            temp += derative_drivetrain(P, x, u);
            temp += derative_tire_model(P, x, u);
            temp += derative_wheels_dynamics_model(P, x, u);
        
            return temp;
        }

   
    // fukncja do liczenia loga przydatnych rzeczy z informacji o stanie pojazdu w danym kroku symulacji


    Log_Info_full log_info_full(const State& x, const Input& u, const ParamBank& P , int step_number){

        Log_Info_full info;


        double m = P.get("m");
        double g = P.get("g");
        double w = P.get("w");
        double a = P.get("a");
        double b = P.get("b");
        double t_front = P.get("t_front");
        double t_rear = P.get("t_rear");
        double h = P.get("h");
        double h_roll_f =  P.get("h1_roll");
        double h_roll_r =  P.get("h2_roll");

        double Kf = P.get("K1");
        double Kr = P.get("K2");
        double K_total =  Kf + Kr;
        double mf = m * a/w;
        double mr = m * b/w;
        double h_prim_f = h - h_roll_f;
        double h_prim_r = h - h_roll_r;
        double epsilon = P.get("epsilon");


        double r_rear = P.get("r_rear");
        double r_front = P.get("r_front");
        double angle_construction_front = P.get("angle_construction_front");
        double angle_construction_rear = P.get("angle_construction_rear");

        double vx_rr = x.vx + x.yaw_rate * r_rear*std::sin(angle_construction_rear);
        double vy_rr = x.vy - x.yaw_rate * r_rear*std::cos(angle_construction_rear);
    
        double vx_rl = x.vx - x.yaw_rate * r_rear*std::sin(angle_construction_rear);
        double vy_rl = x.vy - x.yaw_rate * r_rear*std::cos(angle_construction_rear);
    
        double vx_fr = x.vx + x.yaw_rate * r_front*std::sin(angle_construction_front);
        double vy_fr = x.vy + x.yaw_rate * r_front*std::cos(angle_construction_front);
    
        double vx_fl = x.vx - x.yaw_rate * r_front*std::sin(angle_construction_front);
        double vy_fl = x.vy + x.yaw_rate * r_front*std::cos(angle_construction_front);
    
        // usuwanie nieregularności przy małych prędkościach w rachunkach slipów - > niefizyczne tylko numeryczne : https://www.amazon.pl/Tire-Vehicle-Dynamics-Hans-Pacejka/dp/0080970168 strona z defincją Magic Fomrula
     
       // dodanie symetrczyności wbrew defjincji slipu z MF 5.2 by usunać niefizyczne zachowania przy bardzo małych prędkościach/ odwracaniu momentu obrotowego
       const double vx_rr_denom = std::max(std::abs(vx_rr),1.0);
       const double vx_rl_denom = std::max(std::abs(vx_rl),1.0 );
       const double vx_fr_denom = std::max(std::abs(vx_fr),1.0 );
       const double vx_fl_denom = std::max(std::abs(vx_fl), 1.0);

        double slip_angle_fr = x.delta_right - std::atan2(vy_fr,vx_fr_denom) ;
        double slip_angle_fl = x.delta_left  - std::atan2(vy_fl,vx_fl_denom) ;
        double slip_angle_rr =  - std::atan2(vy_rr,vx_rr_denom);
        double slip_angle_rl =  - std::atan2(vy_rl,vx_rl_denom);

        double slip_ratio_rr = (x.omega_rr * P.get("R") - vx_rr )/vx_rr_denom;
        double slip_ratio_rl = (x.omega_rl * P.get("R") - vx_rl) / vx_rl_denom;
        double slip_ratio_fr = (x.omega_fr * P.get("R") - vx_fr) / vx_fr_denom;
        double slip_ratio_fl = (x.omega_fl * P.get("R") - vx_fl) / vx_fl_denom;

        info.kappa_fl = slip_ratio_fl;
        info.kappa_fr = slip_ratio_fr;
        info.kappa_rl = slip_ratio_rl;
        info.kappa_rr = slip_ratio_rr;

        info.slip_angle_fl = slip_angle_fl * 180 / M_PI ;
        info.slip_angle_fr = slip_angle_fr * 180 / M_PI ;
        info.slip_angle_rl =  slip_angle_rl * 180 / M_PI ;
        info.slip_angle_rr = slip_angle_rr * 180 / M_PI ;
     

        info.slip_angle_body = std::atan2(x.vy, std::max(x.vx,1.0)) * 180 / M_PI ;

        info.fz_fl = 0.5 * mf * g - 0.5 * m * x.prev_ax * h/w  - x.prev_ay/t_front * ( mf * h_roll_f + Kf/K_total *(mf * h_prim_f + mr * h_prim_r)) + 1.0/2 * P.get("Cl1") * x.vx * x.vx ;
        info.fz_fr =   0.5 * mf * g  - 0.5 * m * x.prev_ax * h/w  + x.prev_ay /t_front * ( mf * h_roll_f + Kf/K_total *(mf * h_prim_f + mr * h_prim_r)) + 1.0/2 * P.get("Cl1") * x.vx * x.vx ;
        info.fz_rl =   0.5 * mr * g  + 0.5 * m * x.prev_ax * h/w  - x.prev_ay/t_rear * ( mr * h_roll_r + Kr/K_total *(mf * h_prim_f + mr * h_prim_r)) + 1.0/2 * P.get("Cl2") * x.vx * x.vx ;
        info.fz_rr =  0.5 * mr *g   + 0.5 * m * x.prev_ax * h/w  + x.prev_ay/t_rear * ( mr * h_roll_r + Kr/K_total *(mf * h_prim_f + mr * h_prim_r)) + 1.0/2 * P.get("Cl2") * x.vx * x.vx ;

        info.fy_fl = x.fy_fl;
        info.fy_fr = x.fy_fr;
        info.fy_rl = x.fy_rl;
        info.fy_rr = x.fy_rr;

        const double torque_direct_on_nm  = 3.0;
        const double torque_direct_off_nm = 6.0;
        const double R = P.get("R");

        info.fx_fl = effective_fx_from_torque_blend_(
            x.fx_fl, x.torque_fl, R, torque_direct_on_nm, torque_direct_off_nm);
        info.fx_fr = effective_fx_from_torque_blend_(
            x.fx_fr, x.torque_fr, R, torque_direct_on_nm, torque_direct_off_nm);
        info.fx_rl = effective_fx_from_torque_blend_(
            x.fx_rl, x.torque_rl, R, torque_direct_on_nm, torque_direct_off_nm);
        info.fx_rr = effective_fx_from_torque_blend_(
            x.fx_rr, x.torque_rr, R, torque_direct_on_nm, torque_direct_off_nm);

        info.Power_total = (x.torque_fl * x.omega_fl + x.torque_fr * x.omega_fr + x.torque_rl * x.omega_rl + x.torque_rr * x.omega_rr) / 1000.0;

        info.torque_fl = x.torque_fl;
        info.torque_fr = x.torque_fr;
        info.torque_rl = x.torque_rl;
        info.torque_rr = x.torque_rr;

        info.omega_rl = x.omega_rl;
        info.omega_rr = x.omega_rr;
        info.omega_fl = x.omega_fl;
        info.omega_fr = x.omega_fr;

        info.delta_left = x.delta_left ;
        info.delta_rigth = x.delta_right;
        info.rack_angle = x.rack_angle ;

        info.ax = x.prev_ax; 
        info.ay = x. prev_ay; 

        info.yaw_rate = x.yaw_rate;
        info.vx = x.vx;
        info.vy = x.vy;

        info.time = step_number * P.get("simulation_time_step");

        info.rack_angle_request = u.rack_angle_request ;
        info.torque_request_fl = u.torque_request_fl;
        info.torque_request_fr = u.torque_request_fr;
        info.torque_request_rl = u.torque_request_rl;
        info.torque_request_rr = u.torque_request_rr;
        info.x = x.x;
        info.y = x.y;
        info.yaw = x.yaw;
        info.total_drag = P.get("Cd") * x.vx * x.vx + P.get("Cr") * ( P.get("m") * P.get("g") + P.get("Cl1") * x.vx * x.vx + P.get("Cl2") * x.vx * x.vx ) ;
        info.total_downforce = P.get("Cl1") * x.vx * x.vx + P.get("Cl2") * x.vx * x.vx;

        return info;
}

    void euler_sim_timestep(State& x, const Input& u, const ParamBank& P){
        double dt = P.get("simulation_time_step");
        State dx = model_derative(P,x,u);
        x += dx * dt;
        unwrap_angle(x.yaw);

        // === limits for power and steer (4 wheels) ===

        const double T_min   = P.get("min_torque");    // total (sum)
        const double T_max   = P.get("max_torque");    // total (sum)
        const double P_rec   = P.get("P_min_recup");   // W (może być ujemne)
        const double P_drv   = P.get("P_max_drive");   // W

        // per-wheel (uproszczone 1/4)
        const double T_min_w = 1 * T_min;
        const double T_max_w = 1 * T_max;
        const double P_rec_w = 0.25 * std::abs(P_rec);
        const double P_drv_w = 0.25 * std::abs(P_drv);

        const double omega_eps = 15.0; // rad/s

        auto omega_safe = [&](double w){
            return std::copysign(std::max(std::abs(w), omega_eps), (w == 0.0 ? 1.0 : w));
        };

        auto clampd = [](double v, double lo, double hi){
            return (v < lo) ? lo : ((v > hi) ? hi : v);
        };

        auto limit_wheel_torque = [&](double torque, double omega){
            const double om = omega_safe(omega);

            // moment hard
            const double Tmin_m = T_min_w;
            const double Tmax_m = T_max_w;

            // power (T = P/ω)
            const double Tmin_p = -P_rec_w / om;
            const double Tmax_p =  P_drv_w / om;

            const double Tmin = std::min(Tmin_m, Tmin_p);
            const double Tmax = std::max(Tmax_m, Tmax_p);
            return clampd(torque, Tmin, Tmax);
        };

        x.torque_fl = limit_wheel_torque(x.torque_fl, x.omega_fl);
        x.torque_fr = limit_wheel_torque(x.torque_fr, x.omega_fr);
        x.torque_rl = limit_wheel_torque(x.torque_rl, x.omega_rl);
        x.torque_rr = limit_wheel_torque(x.torque_rr, x.omega_rr);

        if(x.d_rack_angle > P.get("max_steering_angle_rate")){
            x.d_rack_angle = P.get("max_steering_angle_rate");
            x.d_delta_left = P.get("max_steering_angle_rate");
            x.d_delta_right = P.get("max_steering_angle_rate");
        }
        if(x.d_rack_angle < P.get("min_steering_angle_rate")){
            x.d_rack_angle = P.get("min_steering_angle_rate");
            x.d_delta_left = P.get("min_steering_angle_rate");
            x.d_delta_right = P.get("min_steering_angle_rate");
        }

        if(x.rack_angle >= P.get("max_steer")){
            x.d_delta_left = std::min(0.0,x.d_delta_left);
            x.d_delta_right = std::min(0.0,x.d_delta_right) ;
            x.d_rack_angle = std::min(0.0,x.d_rack_angle);
        }
        if(x.rack_angle <= P.get("min_steer")){
            x.d_delta_left = std::max(0.0,x.d_delta_left);
            x.d_delta_right = std::max(0.0,x.d_delta_right) ;
            x.d_rack_angle = std::max(0.0,x.d_rack_angle);
        }


        x.rack_angle = std::clamp(x.rack_angle , P.get("min_steer"), P.get("max_steer"));
    }
    // void rk4_sim_timestep(State& x, const Input& u, const ParamBank& P){
    //     double dt = P.get("simulation_time_step");
    //     State k1 = model_derative(P,x,u);
    //     State k2 = model_derative(P,x + 0.5 * dt * k1,u);
    //     State k3 = model_derative(P,x + 0.5 * dt * k2,u);
    //     State k4 = model_derative(P,x + dt * k3,u);
    //     x += (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    //     unwrap_angle(x.yaw);
    // }

}
