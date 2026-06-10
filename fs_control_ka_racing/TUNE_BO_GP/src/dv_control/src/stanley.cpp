#include "stanley.hpp"

namespace v2_control {

Stanley::Stanley(){
   std::cout<<" deafult initiliazer not everything is correct"<<std::endl;
}

Stanley::Stanley( const ParamBank &P){
	
	// inclizacja na zero -> by nie wyrzuciło na dzień dobry głupot
	cross_track_error = 0 ;
	path_yaw = 0;
	angle_error = 0;
    lookAheadPointInGlobalFrame = Vec2(0,0);
	
	// wczyranie parametrów
	
	param_ = P;
}


void Stanley::setTrack(const Eigen::VectorXd &X, const Eigen::VectorXd &Y)
{
    if (X.size() != Y.size()) {
        ROS_WARN_STREAM("[Stanley] Rozmiary wektorów X (" << X.size()
                         << ") i Y (" << Y.size()
                         << ") nie są równe. Ścieżka NIE została zaktualizowana.");
        return;   // nie zmieniam ptrack_
    }

    if (X.size() < 2) {
        ROS_WARN_STREAM("[Stanley] Za mało punktów na ścieżce ("
                         << X.size() << "). Funkcja przerwana.");
        return;
    }

    ptrack_.X = X;
    ptrack_.Y = Y;
    ptrack_.n_points = static_cast<int>(X.size());
}


double Stanley::computeSteeringAngle(const State &bolid_state) 
{   

    
    //return angle_error + std::atan2(param_.k * cross_track_error, bolid_state.vx + param_.epsilon);

    double v = std::max(bolid_state.vx,param_.get("v_target"));
    return (angle_error + std::atan2(param_.get("stanley_k") * cross_track_error, v + param_.get("stanley_epsilon")));

}
void Stanley::findLookAheadPoint(const State &bolid_state)
{
    
    
   
    if (ptrack_.n_points < 2) {
        ROS_WARN_STREAM("[Stanley] Za mało punktów na ścieżce ("
                         << ptrack_.n_points << "). Funkcja przerwana.");
        return;
    }
    double bolide_yaw = bolid_state.yaw;
   
    unwrap_angle(bolide_yaw);
   
    int idx_start = 0; // index of the closest to front axle  point on the track
    float min_dist_sqr = std::numeric_limits<float>::max(); // 
    // finding the point closest to front axle of a bolid

    double X_front = bolid_state.X + param_.get("stanley_lf") * std::cos(bolide_yaw);
    double Y_front = bolid_state.Y + param_.get("stanley_lf") * std::sin(bolide_yaw);


    for( int idx = 0; idx < ptrack_.n_points; idx ++)
    {

        double dist_sqr = (X_front - ptrack_.X(idx))*(X_front - ptrack_.X(idx)) + (Y_front - ptrack_.Y(idx))*(Y_front - ptrack_.Y(idx));
         if ( dist_sqr < min_dist_sqr)
        {
        	idx_start = idx;
        	min_dist_sqr = dist_sqr;
        }
    }
  
    
      // Jeżeli najbliższy punkt jest ostatni -> błąd wcześniej w piplinie, nie wyliczymy kąta odcinka -> wyjdź bez zmian.
      if (idx_start >= ptrack_.n_points - 1) {
        ROS_WARN_STREAM("[Stanley] Najbliższy punkt to OSTATNI na ścieżce (idx="
                        << idx_start << "). Brak kolejnego punktu – path_yaw NIE zostanie policzony.");
        return;
    }

    
    lookAheadPointInGlobalFrame.x = ptrack_.X(idx_start);
    lookAheadPointInGlobalFrame.y= ptrack_.Y(idx_start);


    double dx = lookAheadPointInGlobalFrame.x - X_front;
    double dy = lookAheadPointInGlobalFrame.y - Y_front;

    cross_track_error = -std::sin(bolide_yaw) * dx + std::cos(bolide_yaw) * dy;

   

    path_yaw = std::atan2(ptrack_.Y(idx_start+1) - ptrack_.Y(idx_start), ptrack_.X(idx_start+1) - ptrack_.X(idx_start));
    angle_error = path_yaw - bolide_yaw ;
   
    unwrap_angle(angle_error);
    

}

geo_control_return Stanley::StanleyControl(const State &bolid_state) 
{
    findLookAheadPoint(bolid_state);
    double steering_angle = computeSteeringAngle(bolid_state);
    
    geo_control_return control_output;
    control_output.look_ahead_point = lookAheadPointInGlobalFrame;
    control_output.steering_angle = steering_angle;
    
    return control_output;
}

geo_control_return Stanley::StanleyControl(const State &bolid_state, double ey, double path_yaw) 
{   
    // używamy ey i path_yaw z argumentów zamiast z przeszukania, ale nadal aktualizujemy lookAheadPointInGlobalFrame na podstawie bolid_state i ptrack_ (jeśli jest wystarczająco dużo punktów
    cross_track_error = ey;
    this->path_yaw = path_yaw;
    angle_error =  bolid_state.yaw - path_yaw;
    unwrap_angle(angle_error);
    double steering_angle = computeSteeringAngle(bolid_state);
    geo_control_return control_output;
    findLookAheadPoint(bolid_state); // Aktualizujemy lookAheadPointInGlobalFrame, ale możemy go potem nadpisać
    control_output.look_ahead_point = lookAheadPointInGlobalFrame; // Możesz też ustawić to na jakiś inny punkt, jeśli chcesz
    control_output.steering_angle = steering_angle;
    
    return control_output;
}
}