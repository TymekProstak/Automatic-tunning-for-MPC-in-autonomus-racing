#include "sim_loop.hpp"
#include <ros/ros.h>

using namespace lem_dynamics_sim_;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lem_simulation_node");
    ros::NodeHandle nh("~");

    // ====== 1) Pobranie parametrów z launch'a ======
    std::string param_file, cones_file, log_file;
    bool logging_enabled_param;

    if (!nh.getParam("param_file", param_file))
        throw std::runtime_error("Missing required param: param_file");

    if (!nh.getParam("cones_file", cones_file))
        throw std::runtime_error("Missing required param: cones_file");

    if (!nh.getParam("log_file", log_file))
        throw std::runtime_error("Missing required param: log_file");


    ROS_INFO_STREAM("LEM Simulation node starting...");
    ROS_INFO_STREAM("  → param_file = " << param_file);
    ROS_INFO_STREAM("  → track_file = " << cones_file);
    ROS_INFO_STREAM("  → log_file   = " << log_file);

    // ====== 2) Inicjalizacja symulatora ======
    Simulation_lem_ros_node sim(
        nh,
        param_file,
        cones_file,
        log_file
    );

    // ====== 3) Pętla symulacji + callbacki ROS w tym samym wątku ======
    const double loop_hz = 1000.0;     // 1kHz
    ros::Rate rate(loop_hz);

    while (ros::ok())
    {
        ros::spinOnce();   // callbacki tutaj (TEN SAM WĄTEK)
        sim.step();        // jeden krok symulacji

        rate.sleep();
    }

    ROS_INFO("LEM Simulation node stopped gracefully.");
    return 0;
}
