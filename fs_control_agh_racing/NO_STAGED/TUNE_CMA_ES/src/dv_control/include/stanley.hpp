#pragma once

#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include "Vec2.hpp"
#include "utilities.hpp"
#include "ParamBank.hpp"

namespace v2_control{

  using json = nlohmann::json;


class Stanley{


  public:
 
  Stanley();
  Stanley(const  ParamBank &P);
  void setTrack(const Eigen::VectorXd &X, const Eigen::VectorXd &Y);
  Track_data ptrack_;
  
  inline double get_path_yaw() const {return path_yaw;}
  inline double get_angle_error() const {return angle_error;}
  inline double get_cross_track_error() const { return cross_track_error;}
  geo_control_return StanleyControl(const State &bolid_state) ;
  geo_control_return StanleyControl(const State &bolid_state, double ey, double path_yaw) ;


  private:
  
   Vec2 lookAheadPointInGlobalFrame;
   double cross_track_error;
   double path_yaw; 
   double angle_error;

  ParamBank param_;
   void findLookAheadPoint(const State &bolid_state);
   double computeSteeringAngle(const State &bolid_state)  ;

    
};
}
