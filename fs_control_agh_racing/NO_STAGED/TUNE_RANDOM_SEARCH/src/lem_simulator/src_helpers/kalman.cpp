#include "kalman.hpp"
#include <cmath>

namespace lem_dynamics_sim_ {

static inline double sqr(double x) { return x * x; }

void KalmanFilter::predict(const ImuMeasurement& imu_meas, double dt)
{
    last_imu_meas_ = imu_meas;
    if (!is_initialized_) {
        return;
    }

    // ---------------------------
    // 1) NOMINAL STATE PROPAGATION
    // ---------------------------
    const double yaw0 = x_(YAW);

    // Save old velocity for position integration
    const double vx0 = x_(VX);
    const double vy0 = x_(VY);

    // Stałe biasy (nie-estymowane) — najprościej 0.0
    // Jeśli chcesz, możesz wziąć z param_.get("acc_bias_x") itd.
    const double b_ax   = 0.0;
    const double b_ay   = 0.0;
    const double b_gyro = 0.0;

    // Bias-correct IMU using constants
    const double ax_b = imu_meas.ax       - b_ax;
    const double ay_b = imu_meas.ay       - b_ay;
    const double wz   = imu_meas.yaw_rate - b_gyro;

    const double c = std::cos(yaw0);
    const double s = std::sin(yaw0);

    // Body -> world acceleration
    const double ax_w =  c * ax_b - s * ay_b;
    const double ay_w =  s * ax_b + c * ay_b;

    // Yaw
    double yaw1 = yaw0 + wz * dt;
    unwrap_angle(yaw1);
    x_(YAW) = yaw1;

    // Velocity (world)
    x_(VX) = vx0 + ax_w * dt;
    x_(VY) = vy0 + ay_w * dt;

    // Position (world): use old velocity + 0.5*a*dt^2
    x_(X) = x_(X) + vx0 * dt + 0.5 * ax_w * dt * dt;
    x_(Y) = x_(Y) + vy0 * dt + 0.5 * ay_w * dt * dt;

    // ---------------------------
    // 2) COVARIANCE PROPAGATION (EKF/ESKF-lite without biases)
    // ---------------------------
    // State: [x, y, yaw, vx, vy]
    // Noise w: [n_ax, n_ay, n_g]  (IMU white noise PER-SAMPLE)
    //
    // Convention:
    // - acc_noise: [m/s^2] per-sample std
    // - gyro_noise: [rad/s] per-sample std

    MatN F = MatN::Identity();
    Eigen::Matrix<double, 5, 3> G = Eigen::Matrix<double, 5, 3>::Zero();

    // δp = δp + δv*dt
    F(X, VX) = dt;
    F(Y, VY) = dt;

    // d(R a_b)/dψ (computed at yaw0, with a_b = [ax_b; ay_b])
    // d(R a_b)/dψ = [ -s -c; c -s ] * [ax_b; ay_b]
    const double d_ax_dpsi = -(s * ax_b + c * ay_b);
    const double d_ay_dpsi =  (c * ax_b - s * ay_b);

    // δyaw affected by gyro noise: δyaw += n_g * dt
    // (no bias term)
    // v affected by δyaw: δv += (d a_w/dψ)*δψ*dt
    F(VX, YAW) = d_ax_dpsi * dt;
    F(VY, YAW) = d_ay_dpsi * dt;

    // Position also depends on acceleration: 0.5*a*dt^2
    F(X, YAW) = 0.5 * d_ax_dpsi * dt * dt;
    F(Y, YAW) = 0.5 * d_ay_dpsi * dt * dt;

    // IMU per-sample accel noise -> affects v via dt and p via 0.5*dt^2 (mapped by R)
    G(VX, 0) =  c * dt;
    G(VX, 1) = -s * dt;
    G(VY, 0) =  s * dt;
    G(VY, 1) =  c * dt;

    G(X, 0)  =  0.5 * c * dt * dt;
    G(X, 1)  = -0.5 * s * dt * dt;
    G(Y, 0)  =  0.5 * s * dt * dt;
    G(Y, 1)  =  0.5 * c * dt * dt;

    // gyro noise -> yaw via dt
    G(YAW, 2) = dt;

    // Params
    const double sig_a_step = param_.get("acc_noise_std");   // [m/s^2] per-sample std
    const double sig_g_step = param_.get("gyro_noise_std");  // [rad/s] per-sample std

    Eigen::Matrix<double, 3, 3> Qw = Eigen::Matrix<double, 3, 3>::Zero();
    Qw(0,0) = sqr(sig_a_step);
    Qw(1,1) = sqr(sig_a_step);
    Qw(2,2) = sqr(sig_g_step);

    // Propagate covariance
    P_ = F * P_ * F.transpose() + G * Qw * G.transpose();

    // Symmetrize
    P_ = 0.5 * (P_ + P_.transpose());
}

void KalmanFilter::update_gps(const GpsMeasurement& gps_meas)
{
    if (!is_initialized_) {
        initialize(gps_meas);
        return;
    }

    last_gps_meas_ = gps_meas;

    // z = [x, y, yaw, vx, vy]  (WORLD)
    Eigen::Matrix<double, 5, 1> z;
    z << gps_meas.x, gps_meas.y, gps_meas.yaw, gps_meas.vx, gps_meas.vy;

    Eigen::Matrix<double, 5, 1> hx;
    hx << x_(X), x_(Y), x_(YAW), x_(VX), x_(VY);

    Eigen::Matrix<double, 5, 1> r = z - hx;
    double yaw_res = r(2);
    unwrap_angle(yaw_res);
    r(2) = yaw_res;

    // H = I (bo mierzysz dokładnie stan)
    Eigen::Matrix<double, 5, 5> H = Eigen::Matrix<double, 5, 5>::Identity();

    // R
    const double sig_pos = param_.get("gps_position_noise");         // [m]
    const double sig_yaw = param_.get("gps_yaw_noise");  // [rad]
    const double sig_vel = param_.get("gps_speed_noise");   // [m/s]

    Eigen::Matrix<double, 5, 5> R = Eigen::Matrix<double, 5, 5>::Zero();
    R(0,0) = sqr(sig_pos);
    R(1,1) = sqr(sig_pos);
    R(2,2) = sqr(sig_yaw);
    R(3,3) = sqr(sig_vel);
    R(4,4) = sqr(sig_vel);

    // S = HPH^T + R = P + R (bo H=I)
    Eigen::Matrix<double, 5, 5> S = P_ + R;

    // K = P H^T S^{-1} = P S^{-1}
    Eigen::LDLT<Eigen::Matrix<double, 5, 5>> ldlt(S);
    const Eigen::Matrix<double, 5, 5> I5 = Eigen::Matrix<double, 5, 5>::Identity();
    Eigen::Matrix<double, 5, 5> K = P_ * ldlt.solve(I5);

    // Correction
    VecN dx = K * r;

    x_(X)   += dx(X);
    x_(Y)   += dx(Y);

    double yaw_new = x_(YAW) + dx(YAW);
    unwrap_angle(yaw_new);
    x_(YAW) = yaw_new;

    x_(VX)  += dx(VX);
    x_(VY)  += dx(VY);

    // Joseph form: P = (I-KH)P(I-KH)^T + KRK^T
    // but H=I => (I-K)
    Eigen::Matrix<double, 5, 5> I = Eigen::Matrix<double, 5, 5>::Identity();
    P_ = (I - K) * P_ * (I - K).transpose() + K * R * K.transpose();

    // Symmetrize
    P_ = 0.5 * (P_ + P_.transpose());
}

void KalmanFilter::initialize(const GpsMeasurement& gps_meas)
{
    x_.setZero();
    x_(X)   = gps_meas.x;
    x_(Y)   = gps_meas.y;
    x_(YAW) = gps_meas.yaw;
    x_(VX)  = gps_meas.vx;
    x_(VY)  = gps_meas.vy;

    // P0
    double sig_pos = param_.get("gps_position_noise");
    double sig_yaw = param_.get("gps_yaw_noise");
    double sig_vel = param_.get("gps_speed_noise");

    const double k = 2.0; // conservative
    sig_pos *= k; sig_yaw *= k; sig_vel *= k;

    P_.setZero();
    P_(X, X)     = sqr(sig_pos);
    P_(Y, Y)     = sqr(sig_pos);
    P_(YAW, YAW) = sqr(sig_yaw);
    P_(VX, VX)   = sqr(sig_vel);
    P_(VY, VY)   = sqr(sig_vel);

    last_gps_meas_ = gps_meas;
    last_imu_meas_ = ImuMeasurement{0.0, 0.0, 0.0};
    is_initialized_ = true;
}

void KalmanFilter::reset()
{
    is_initialized_ = false;
    x_.setZero();
    P_.setZero();
    last_imu_meas_ = ImuMeasurement{0.0, 0.0, 0.0};
    last_gps_meas_ = GpsMeasurement{0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace lem_dynamics_sim_