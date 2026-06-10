#include <ros/ros.h>
#include <dv_interfaces/Control.h>
#include <nav_msgs/Odometry.h>
#include <cmath>
#include <algorithm>

double current_vx = 0.0;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // Uproszczone pobranie prędkości wzdłużnej (zakładamy jazdę na wprost)
    current_vx = msg->twist.twist.linear.x; 
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "false_dv_control_node");
    ros::NodeHandle nh;

    ros::Publisher control_pub = nh.advertise<dv_interfaces::Control>("/dv_board/control", 1);
    ros::Subscriber odom_sub = nh.subscribe("/ins/pose", 1, odomCallback);

    ros::Rate rate(40); // 40 Hz
    double amplitude = 2.0; // ax_target od -7 do 7 m/s^2
    double frequency = 0.2; // 0.2 Hz (okres 5s)

    // Parametry pojazdu (z control_param.json)
    const double m = 193.0 + 14.0 ;
    const double Cd = 0.814;
    const double Cr0 = 25.0;
    const double wheel_radius = 0.195;
    const double max_motor_torque = 250.3;

    while (ros::ok()) {
        ros::spinOnce();

        double t = ros::Time::now().toSec();
        //double ax_target = amplitude * std::sin(2.0 * M_PI * frequency * t) + 3.0; // Dodaj offset, aby mieć głównie przyspieszanie
        double ax_target = 20.0;
        // Obliczenie siły wzdłużnej (fx_target) i momentu
        double drag_force = Cd * current_vx * current_vx + Cr0;
        double F_required = m * (ax_target) + drag_force;
        double torque_required = F_required * wheel_radius;
        double throttle_command = (torque_required / (4.0 * max_motor_torque)) * 100.0;

        // Ograniczenie throttle
        throttle_command =  100.0;

        dv_interfaces::Control msg;
        msg.move_type = dv_interfaces::Control::TORQUE_PERCENTAGE;
        msg.movement = static_cast<float>(throttle_command);
        msg.steeringAngle_rad = 0.0;
        msg.serviceBrake = 0;
        msg.finished = false;

        msg.fx_target = static_cast<float>(F_required);
        msg.mz_target = 0.0;
        msg.ax_target = static_cast<float>(ax_target);
        msg.next_yaw_rate_target = 0.0;
        msg.current_speed = static_cast<float>(current_vx);

        control_pub.publish(msg);

        rate.sleep();
    }

    return 0;
}