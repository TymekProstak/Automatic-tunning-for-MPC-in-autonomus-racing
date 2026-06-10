#pragma once

#include "uttilities.hpp"

namespace lem_dynamics_sim_{

    struct Track_cone {
        double x; // x position in global frame , or when used as a result of function track_in_camera_frame,  wr to camera
        double y; //y  position in global frame , or when used as a result of function track_in_camera_frame,  wr to camera
        double z; // z  position in global frame always 0  or when used as a result of function track_in_camera_frame,  wr to camera
        double distance; // non zero only if when track is considerd with respect to camera
        std::string color; // eg yellow , blue i tak musi być inaczej się slam zesra bo tam jest dict i to są klucze
        

        // Przeciążenie operatora ==
        bool operator==(const Track_cone& other) const {
            return x == other.x && y == other.y && z == other.z && color == other.color;
        }
    };
    

    struct Track {
        std::vector<Track_cone> cones;
        
    
    };

    inline double fast_sqrt(double x) noexcept {
        // 1 iteracja Newtona, wystarczająca do błędu <0.3%
        union { double f; uint64_t i; } u = { x };
        u.i = (u.i + 0x3ff0000000000000ULL) >> 1;
        return 0.5 * (u.f + x / u.f);
    }
    inline double fast_exp(double x) noexcept {
        // Źródło: Schraudolph (1999) - "A fast, compact approximation of the exponential function"
        x = 1.0 + x / 1024.0;
        x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x; x *= x;
        return x;
    }
    
    

    Track load_track_from_csv(const std::string& filename); 
    std::vector<std::vector<std::string>> read_csv(const std::string& filename);

    
} // namespace lem_dynamics_sim_