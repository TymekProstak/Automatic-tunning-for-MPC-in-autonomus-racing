#pragma once

#include "ParamBank.hpp"
#include"uttilities.hpp"
#include "cone_track.hpp"
#include <random>
#include <algorithm>
#include <cmath>


namespace lem_dynamics_sim_{


       Track track_in_camera_frame( const State& state, const Track& track_global , const ParamBank& P);
      

        Track shoot_a_frame(const Track&global_track, const ParamBank& P, const State& state);

} // lem_dynamics_sim_