#include "cone_detector.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>  // std::move

namespace lem_dynamics_sim_ {

//// ===========================================================================
//// 1) Transformacja toru globalnego do układu kamery
//// ===========================================================================

Track track_in_camera_frame(const State& state,
                            const Track& track_global,
                            const ParamBank& P)
{
    Track track_vehicle_frame;
    track_vehicle_frame.cones.reserve(track_global.cones.size());

    const double cos_yaw = std::cos(state.yaw);
    const double sin_yaw = std::sin(state.yaw);

    // Prefetch offsetów kamery
    const double x_cam = P.get("x_camera_to_cog");
    const double y_cam = P.get("y_camera_to_cog");
    const double z_cam = P.get("z_camera_to_cog");

    for (const auto& cone : track_global.cones)
    {
        const double dx = cone.x - state.x;
        const double dy = cone.y - state.y;

        Track_cone cv;
        cv.x = dx * cos_yaw + dy * sin_yaw + x_cam;
        cv.y = -dx * sin_yaw + dy * cos_yaw + y_cam;
        cv.z = cone.z + z_cam;
        cv.color = cone.color;

        const double dist = std::sqrt(cv.x * cv.x + cv.y * cv.y + cv.z * cv.z);
        cv.distance = dist;

        track_vehicle_frame.cones.emplace_back(std::move(cv));
    }

    return track_vehicle_frame;
}

//// ===========================================================================
//// 2) Symulacja pojedynczego zdjęcia („shoot_a_frame”)
////    – filtracja stożków w FOV, model wykrycia, szum pozycji (Gauss)
//// ===========================================================================

Track shoot_a_frame(const Track& global_track, const ParamBank& P, const State& state)
{
    // --- 1. Transformacja do układu kamery
    Track local = track_in_camera_frame(state, global_track, P);

    // --- 2. Prefetch parametrów
    const double max_range    = P.get("camera_range");
    const double fov_W        = P.get("fov_x") * M_PI / 180.0;
    const double fov_H        = P.get("fov_y") * M_PI / 180.0;
    const double tan_W2       = std::tan(0.5 * fov_W);
    const double tan_H2       = std::tan(0.5 * fov_H);
    const double camera_range = 16.0; // 16 m by tak ogólnie generalnie dobrze działał sytem można dać więcej, slam dostaje idealne cony

   

    // --- 4. Wynikowy tor z widocznymi stożkami
    Track visible;
    visible.cones.reserve(local.cones.size());

    // --- 5. Główna pętla filtracji i symulacji błędu
    for (auto& c : local.cones)
    {
        // (a) Odrzuć stożki za kamerą
        if (c.x <= 0.0) continue;

        // (b) Odrzuć spoza zasięgu
        if (c.distance > max_range) continue;

        // (c) Odrzuć spoza pola widzenia
        if (std::abs(c.y) > tan_W2 * c.x) continue;
        if (std::abs(c.z) > tan_H2 * c.x) continue;

        // wszytskie pachołki widać / zerowy szum 

        // dystansu nie aktualizujemy (spójność z pipeline’em SLAM)
        visible.cones.emplace_back(std::move(c));
    }

    return visible;
}

} // namespace lem_dynamics_sim_
