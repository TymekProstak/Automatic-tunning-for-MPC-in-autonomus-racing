#include <iostream>
#include <chrono>
#include <random>
#include "cone_detector.hpp"
#include "uttilities.hpp"
#include "ParamBank.hpp"

using namespace lem_dynamics_sim_;

int main() {
    // Inicjalizacja parametrów
    ParamBank P;
    P.add("x_cog_to_camera", 0.0);
    P.add("y_cog_to_camera", 0.0);
    P.add("z_cog_to_camera", 0.5);
    P.add("fov_W", 90.0 * M_PI / 180.0);  // 90 stopni w radianach
    P.add("fov_H", 60.0 * M_PI / 180.0);  // 60 stopni w radianach
    P.add("max_vision_range", 50.0);
    P.add("camera_range", 50.0);

    // Inicjalizacja stanu bolidu
    State state;
    state.x = 0.0;
    state.y = 0.0;
    state.yaw = 0.0;

    // Wygenerowanie losowych stożków
    Track global_track;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-50.0, 50.0);

    for (int i = 0; i < 2000; ++i) {
        Track_cone cone;
        cone.x = dist(gen);
        cone.y = dist(gen);
        cone.z = 0.0;
        cone.color = (i % 2 == 0) ? "yellow" : "blue";  // Naprzemiennie żółte i niebieskie
        global_track.cones.push_back(cone);
    }

    // Pomiar czasu wykonania funkcji shoot_a_frame
    auto start = std::chrono::high_resolution_clock::now();
    Track visible_track = shoot_a_frame(global_track, P, state);
    auto end = std::chrono::high_resolution_clock::now();

    // Wyświetlenie wyników
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Czas wykonania shoot_a_frame: " << elapsed.count() << " sekund" << std::endl;
    std::cout << "Liczba wykrytych stożków: " << visible_track.cones.size() << std::endl;

    return 0;
}