#pragma once

#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<vector>
#include<Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include "Vec2.hpp"
#include <ros/ros.h>


namespace v2_control{

  using json = nlohmann::json;

struct Track_data{

 Eigen::VectorXd X;
 Eigen::VectorXd Y;
 int n_points;

};

struct State{
 double X;
 double Y;
 double yaw;
 double delta;
 double delta_dot;
 double vx;
 double vy;
 double yaw_rate;
 double acc;
};

inline double unwrap_angle(double &angle){

    while (angle > M_PI)
    {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI)
    {
        angle += 2.0 * M_PI;
    }
    return angle;
}
struct geo_control_return{
  Vec2 look_ahead_point;
  double steering_angle;
};


}