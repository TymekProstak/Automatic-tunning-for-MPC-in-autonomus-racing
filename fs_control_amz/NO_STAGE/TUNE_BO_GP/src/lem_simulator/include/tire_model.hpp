#pragma once

#include "ParamBank.hpp"
#include"uttilities.hpp"

namespace lem_dynamics_sim_{
    
    State derative_tire_model( const ParamBank& P, const State& x, const Input& u) ;

}
