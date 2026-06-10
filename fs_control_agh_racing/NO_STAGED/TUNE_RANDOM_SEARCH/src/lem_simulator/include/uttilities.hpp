#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

namespace lem_dynamics_sim_ {

struct State {
    double x, y, yaw, vx, vy, yaw_rate;
    // 4 wheel angular speeds
    double omega_rr, omega_rl, omega_fr, omega_fl;
    double delta_left, d_delta_left, delta_right, d_delta_right;
    double rack_angle, d_rack_angle;

    // torques per wheel (no aggregate torque)
    double torque_rr, torque_rl, torque_fr, torque_fl;

    // longitudinal + lateral tire forces per wheel
    double fx_rr, fx_rl, fx_fr, fx_fl;
    double fy_rr, fy_rl, fy_fr, fy_fl;

    double prev_ax, prev_ay;

    State();
    explicit State(double value);
    explicit State(const std::vector<double>& values);

    void setZero();

    State operator+(const State& other) const;
    State& operator+=(const State& other);
    State operator*(double scalar) const;
    friend State operator*(double scalar, const State& s);
};

struct Input {
    // 4 wheel torque requests + steering
    double torque_request_rr, torque_request_rl, torque_request_fr, torque_request_fl;
    double rack_angle_request;

    Input();
    Input(double torque_rr, double torque_rl, double torque_fr, double torque_fl, double rack);
    explicit Input(double value);
    explicit Input(const std::vector<double>& values);
};

void unwrap_angle(double& angle);

} // namespace lem_dynamics_sim_
