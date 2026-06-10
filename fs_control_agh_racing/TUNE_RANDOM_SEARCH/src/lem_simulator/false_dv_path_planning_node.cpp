#include <ros/ros.h>
#include <dv_interfaces/LtoResult.h>

#include <Eigen/Dense>

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "ParamBank_lto.hpp"
#include "spline.hpp"      // TrackSpline2D
#include "Vec2.hpp"
namespace {

struct XY { double x{}, y{}; };

static bool parseTrackCsv(const std::string& file_path,
                          std::vector<XY>& out_points,
                          std::string& err)
{
  out_points.clear();

  std::ifstream in(file_path);
  if (!in.is_open()) {
    err = "cannot open file: " + file_path;
    return false;
  }

  std::string line;
  bool any_data = false;

  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

    // accept "x,y" or ";" and tolerate extra cols
    std::string normalized = line;
    for (char& c : normalized) if (c == ';') c = ',';

    std::stringstream ss(normalized);
    std::string token_x, token_y;
    if (!std::getline(ss, token_x, ',')) continue;
    if (!std::getline(ss, token_y, ',')) continue;

    try {
      double x = std::stod(token_x);
      double y = std::stod(token_y);
      out_points.push_back({x, y});
      any_data = true;
    } catch (...) {
      continue; // header / malformed
    }
  }

  if (!any_data || out_points.size() < 3) {
    err = "track csv has too few valid points (<3): " + file_path;
    return false;
  }
  return true;
}

static inline void eigenFromXY(const std::vector<XY>& pts, Eigen::VectorXd& X, Eigen::VectorXd& Y)
{
  const int N = static_cast<int>(pts.size());
  X.resize(N);
  Y.resize(N);
  for (int i = 0; i < N; ++i) {
    X[i] = pts[i].x;
    Y[i] = pts[i].y;
  }
}

// --- overloady: jeśli LTO trzyma Eigen::VectorXd albo std::vector<double> ---
static inline std::vector<double> toStdVec(const Eigen::VectorXd& v)
{
  return std::vector<double>(v.data(), v.data() + v.size());
}
static inline std::vector<double> toStdVec(const std::vector<double>& v)
{
  return v;
}

static bool loadJsonFile(const std::string& path, nlohmann::json& J, std::string& err)
{
  std::ifstream f(path);
  if (!f.is_open()) {
    err = "cannot open json file: " + path;
    return false;
  }
  try {
    f >> J;
  } catch (const std::exception& e) {
    err = std::string("json parse error: ") + e.what();
    return false;
  }
  return true;
}

// długość polilinii z domknięciem (track jest pętlą bez duplikatu końca)
static double polylineLengthClosed(const std::vector<double>& xs, const std::vector<double>& ys)
{
  const size_t N = std::min(xs.size(), ys.size());
  if (N < 2) return 0.0;

  double L = 0.0;
  for (size_t i = 1; i < N; ++i) {
    const double dx = xs[i] - xs[i - 1];
    const double dy = ys[i] - ys[i - 1];
    L += std::sqrt(dx*dx + dy*dy);
  }

  // domknięcie
  const double dx = xs.front() - xs.back();
  const double dy = ys.front() - ys.back();
  L += std::sqrt(dx*dx + dy*dy);

  return L;
}

static inline bool isFinite(double x) { return std::isfinite(x); }

} // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "false_lto_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Publikujemy raz, ale latched (nowy sub dostanie od razu)
  ros::Publisher lto_pub = nh.advertise<dv_interfaces::LtoResult>(
      "/path_planning/path", 1, true /* latch */);

  // --- params ---
  std::string track_file;
  if (!pnh.getParam("track_file", track_file)) (void)nh.getParam("track_file", track_file);
  if (track_file.empty()) {
    ROS_ERROR_STREAM("[false_lto_node] Missing param 'track_file' (CSV x,y centerline).");
    return 1;
  }

  std::string param_file;
  if (!pnh.getParam("param_file", param_file)) (void)nh.getParam("param_file", param_file);
  if (param_file.empty()) {
    ROS_ERROR_STREAM("[false_lto_node] Missing param 'param_file' (control_params.json).");
    return 1;
  }

  bool racing_line = false;
  pnh.param("racing_line", racing_line, false);

  std::string racing_line_file;
  if (!pnh.getParam("racing_line_file", racing_line_file)) (void)nh.getParam("racing_line_file", racing_line_file);

  // 1) Load CSV (centerline or racing line)
  std::vector<XY> pts;
  std::string err;
  std::string file_to_load = racing_line ? racing_line_file : track_file;
  
  if (!parseTrackCsv(file_to_load, pts, err)) {
    ROS_ERROR_STREAM("[false_lto_node] Failed to load track: " << err);
    return 1;
  }

  Eigen::VectorXd X_path, Y_path;
  eigenFromXY(pts, X_path, Y_path);

  // 2) Load ParamBank from JSON
  nlohmann::json J;
  if (!loadJsonFile(param_file, J, err)) {
    ROS_ERROR_STREAM("[false_lto_node] Failed to load params: " << err);
    return 1;
  }

  lto::ParamBank_lto P;
  try {
    P = lto::build_param_bank(J);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("[false_lto_node] build_param_bank threw: " << e.what());
    return 1;
  }

 
    dv_interfaces::LtoResult msg;
    
    const int N_pts = pts.size();
    std::vector<double> track_x(N_pts);
    std::vector<double> track_y(N_pts);
    std::vector<double> zeros(N_pts, 0.0);
    
    for (int i = 0; i < N_pts; ++i) {
      track_x[i] = pts[i].x;
      track_y[i] = pts[i].y;
    }
    
    msg.track_x = track_x;
    msg.track_y = track_y;
    msg.racing_line = racing_line;
    
    lto_pub.publish(msg);
  

  ros::spin();
  return 0;
}
