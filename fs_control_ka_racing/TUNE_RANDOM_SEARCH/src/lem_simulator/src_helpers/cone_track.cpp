#include "cone_track.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cctype>     // std::isspace
#include <cstdlib>    // std::strtod

namespace lem_dynamics_sim_
{
    // --- helper: trim whitespace ---
    static inline std::string trim_copy(const std::string& s)
    {
        size_t b = 0;
        while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
        size_t e = s.size();
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
        return s.substr(b, e - b);
    }

    // --- helper: bezpieczne parsowanie double (ignoruje header) ---
    static inline bool try_parse_double(const std::string& s, double& out)
    {
        std::string t = trim_copy(s);
        if (t.empty()) return false;

        char* end = nullptr;
        const char* c = t.c_str();
        out = std::strtod(c, &end);

        // end == c => nic nie sparsował
        if (end == c) return false;

        // jeżeli po liczbie są jeszcze jakieś nie-spacje -> traktuj jako błąd
        while (*end != '\0') {
            if (!std::isspace(static_cast<unsigned char>(*end))) return false;
            ++end;
        }
        return true;
    }

    std::vector<std::vector<std::string>> read_csv(const std::string& filename)
    {
        std::vector<std::vector<std::string>> data;
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            // pomijam puste linie i komentarze
            std::string tl = trim_copy(line);
            if (tl.empty()) continue;
            if (!tl.empty() && tl[0] == '#') continue;

            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                row.push_back(trim_copy(cell));
            }

            // pomijam wiersze zbyt krótkie
            if (!row.empty()) data.push_back(row);
        }

        return data;
    }

    Track load_track_from_csv(const std::string& filename)
    {
        Track track;
        const auto data = read_csv(filename);

        for (const auto& row : data) {
            if (row.size() < 3) continue;

            double x, y;
            if (!try_parse_double(row[0], x)) continue; // np. "x" z headera
            if (!try_parse_double(row[1], y)) continue; // np. "y" z headera

            Track_cone cone;
            cone.x = x;
            cone.y = y;
            cone.z = 0.0;
            cone.color = row[2];

            track.cones.push_back(cone);
        }

        return track;
    }

} // namespace lem_dynamics_sim_
