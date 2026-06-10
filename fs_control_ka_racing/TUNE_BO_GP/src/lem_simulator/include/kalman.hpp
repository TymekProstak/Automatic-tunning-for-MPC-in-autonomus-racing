#pragma once

#include <Eigen/Dense>

#include "uttilities.hpp"
#include "ParamBank.hpp"

namespace lem_dynamics_sim_ {

struct KalmanState {
    double x;
    double y;
    double yaw;
    double vx;
    double vy;

    // Biasy nie są estymowane – tylko telemetry/debug (stałe z paramów albo 0)
    double b_ax;
    double b_ay;
    double b_gyro;

    double yaw_rate; // convenience: yaw_rate_meas - b_gyro
};

struct ImuMeasurement {
    double ax;
    double ay;
    double yaw_rate;
};

struct GpsMeasurement {
    double x;
    double y;
    double yaw;
    double vx;
    double vy;
};

class KalmanFilter {
public:
    KalmanFilter() = default;

    static constexpr int kN = 5; // [x, y, yaw, vx, vy]
    using VecN = Eigen::Matrix<double, kN, 1>;
    using MatN = Eigen::Matrix<double, kN, kN>;

    explicit KalmanFilter(const ParamBank& P) : param_(P) {
        this->reset();
    }

    void predict(const ImuMeasurement& imu_meas, double dt);
    void update_gps(const GpsMeasurement& gps_meas);

    KalmanState get_state() const {
        return this->pack_state(x_);
    }

    void initialize(const GpsMeasurement& gps_meas);
    void reset();

    ImuMeasurement get_last_imu_measurement() const { return last_imu_meas_; }
    GpsMeasurement get_last_gps_measurement() const { return last_gps_meas_; }

private:
    // Kolejność stanu: [x, y, yaw, vx, vy]
    enum Idx : int { X = 0, Y = 1, YAW = 2, VX = 3, VY = 4 };

    VecN x_{VecN::Zero()};
    MatN P_{MatN::Identity()};

    ParamBank param_;
    bool is_initialized_ = false;

    ImuMeasurement last_imu_meas_{};
    GpsMeasurement last_gps_meas_{};

private:
    KalmanState pack_state(const VecN& x) const {
        KalmanState s{};
        s.x   = x(X);
        s.y   = x(Y);
        s.yaw = x(YAW);
        s.vx  = x(VX);
        s.vy  = x(VY);

        // Stałe biasy (jeśli nie masz w paramach, ustaw na 0)
        const double b_ax   = 0.0;
        const double b_ay   =  0.0;
        const double b_gyro =  0.0;

        s.b_ax   = b_ax;
        s.b_ay   = b_ay;
        s.b_gyro = b_gyro;

        s.yaw_rate = last_imu_meas_.yaw_rate - b_gyro;
        return s;
    }

    VecN unpack_state(const KalmanState& s) const {
        VecN x = VecN::Zero();
        x(X)   = s.x;
        x(Y)   = s.y;
        x(YAW) = s.yaw;
        x(VX)  = s.vx;
        x(VY)  = s.vy;
        return x;
    }
};

} // namespace lem_dynamics_sim_