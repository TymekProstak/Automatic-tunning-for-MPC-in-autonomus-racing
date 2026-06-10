#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "Vec2.hpp"

namespace v2_control {

// ============================================================
//  Common helpers
// ============================================================

struct SplineSegment {
    double a, b, c, d; // y = a + b*dt + c*dt^2 + d*dt^3
};

static inline double segEval(const SplineSegment& s, double dt) {
    return s.a + s.b*dt + s.c*dt*dt + s.d*dt*dt*dt;
}
static inline double segDeriv(const SplineSegment& s, double dt) {
    return s.b + 2.0*s.c*dt + 3.0*s.d*dt*dt;
}
static inline double segDeriv2(const SplineSegment& s, double dt) {
    return 2.0*s.c + 6.0*s.d*dt;
}

static inline double cross2(double ax, double ay, double bx, double by) {
    return ax*by - ay*bx;
}

// ============================================================
//  AKIMA 1D (for open path)
// ============================================================

static std::vector<SplineSegment> computeAkima1D(const std::vector<double>& t,
                                                 const std::vector<double>& y)
{
    const size_t n = t.size();
    std::vector<SplineSegment> segs;
    if (n < 2) return segs;

    const size_t m = n - 1;
    std::vector<double> mvals(m);

    for (size_t j = 0; j < m; ++j) {
        double dt = t[j+1] - t[j];
        mvals[j] = (dt == 0.0) ? 0.0 : (y[j+1] - y[j]) / dt;
    }

    std::vector<double> m_ext(m + 4);
    for (size_t j = 0; j < m; ++j)
        m_ext[j + 2] = mvals[j];

    m_ext[1] = 2.0 * m_ext[2] - m_ext[3];
    m_ext[0] = 2.0 * m_ext[1] - m_ext[2];
    m_ext[m + 2] = 2.0 * m_ext[m + 1] - m_ext[m];
    m_ext[m + 3] = 2.0 * m_ext[m + 2] - m_ext[m + 1];

    std::vector<double> d(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        size_t k = i + 2;
        double w1 = std::abs(m_ext[k+1] - m_ext[k]);
        double w2 = std::abs(m_ext[k-1] - m_ext[k-2]);
        double denom = w1 + w2;

        if (denom > 0.0)
            d[i] = (w1 * m_ext[k-1] + w2 * m_ext[k]) / denom;
        else
            d[i] = 0.5 * (m_ext[k-1] + m_ext[k]);
    }

    segs.resize(m);
    for (size_t i = 0; i < m; ++i) {
        double h = t[i+1] - t[i];
        if (h == 0.0) {
            segs[i] = { y[i], 0.0, 0.0, 0.0 };
            continue;
        }

        double y0 = y[i], y1 = y[i+1];
        double d0 = d[i], d1 = d[i+1];

        double a = y0;
        double b = d0;
        double c = (3*(y1 - y0)/h - 2*d0 - d1) / h;
        double dd = (2*(y0 - y1)/h + d0 + d1) / (h*h);

        segs[i] = { a, b, c, dd };
    }

    return segs;
}

// ============================================================
//  OPEN PATH: Parametric Akima spline 2D (analytic)
// ============================================================

class OpenAkimaSpline2D
{
public:
    void buildOpen(const std::vector<Vec2>& path)
    {
        clear_();

        if (path.size() < 2) return;

        // remove duplicates
        std::vector<Vec2> pts;
        pts.reserve(path.size());
        pts.push_back(path[0]);

        const double eps2 = 1e-12;
        for (size_t i = 1; i < path.size(); ++i) {
            double dx = path[i].x - path[i-1].x;
            double dy = path[i].y - path[i-1].y;
            if (dx*dx + dy*dy > eps2) pts.push_back(path[i]);
        }
        if (pts.size() < 2) return;

        // chord-length t_
        const size_t n = pts.size();
        t_.assign(n, 0.0);

        for (size_t i = 1; i < n; ++i) {
            double dx = pts[i].x - pts[i-1].x;
            double dy = pts[i].y - pts[i-1].y;
            t_[i] = t_[i-1] + std::sqrt(dx*dx + dy*dy);
        }

        totalLen_ = t_.back();
        if (totalLen_ <= 1e-12) return;

        std::vector<double> xs(n), ys(n);
        for (size_t i = 0; i < n; ++i) {
            xs[i] = pts[i].x;
            ys[i] = pts[i].y;
        }

        segX_ = computeAkima1D(t_, xs);
        segY_ = computeAkima1D(t_, ys);

        valid_ = (!segX_.empty() && segX_.size() == segY_.size());
    }

    bool valid() const { return valid_; }
    double totalLength() const { return totalLen_; }

    Vec2 eval(double s) const {
        if (!valid_) return Vec2(0.f, 0.f);
        double sc = clampS_(s);
        int i = findSeg_(sc);
        double dt = sc - t_[i];
        return Vec2((float)segEval(segX_[i], dt),
                    (float)segEval(segY_[i], dt));
    }

    double getX(double s) const {
        if (!valid_) return 0.0;
        double sc = clampS_(s);
        int i = findSeg_(sc);
        double dt = sc - t_[i];
        return segEval(segX_[i], dt);
    }

    double getY(double s) const {
        if (!valid_) return 0.0;
        double sc = clampS_(s);
        int i = findSeg_(sc);
        double dt = sc - t_[i];
        return segEval(segY_[i], dt);
    }

    double getYaw(double s) const {
        if (!valid_) return 0.0;
        double sc = clampS_(s);
        int i = findSeg_(sc);
        double dt = sc - t_[i];
        double dx = segDeriv(segX_[i], dt);
        double dy = segDeriv(segY_[i], dt);
        return std::atan2(dy, dx);
    }

    double getCurvature(double s) const {
        if (!valid_) return 0.0;
        double sc = clampS_(s);
        int i = findSeg_(sc);
        double dt = sc - t_[i];

        double dx  = segDeriv (segX_[i], dt);
        double dy  = segDeriv (segY_[i], dt);
        double ddx = segDeriv2(segX_[i], dt);
        double ddy = segDeriv2(segY_[i], dt);

        double num = dx*ddy - dy*ddx;
        double den = std::pow(dx*dx + dy*dy, 1.5);
        if (den < 1e-12) return 0.0;
        return num / den;
    }

    // s(x,y): lokalna projekcja (clamp), coarse + Newton
    double projectToS(const Vec2& q,
                      double s_init,
                      double search_radius = 3.0,
                      int samples = 41,
                      int newton_iters = 20) const
    {
        if (!valid_) return 0.0;

        double s0 = clampS_(s_init);
        double s_min = std::max(t_.front(), s0 - search_radius);
        double s_max = std::min(t_.back(),  s0 + search_radius);

        double best_s = s0;
        double best_d2 = std::numeric_limits<double>::infinity();

        for (int k = 0; k < samples; ++k) {
            double sc = s_min + (s_max - s_min) * double(k) / double(samples - 1);
            Vec2 p = eval(sc);
            double dx = double(p.x) - double(q.x);
            double dy = double(p.y) - double(q.y);
            double d2 = dx*dx + dy*dy;
            if (d2 < best_d2) { best_d2 = d2; best_s = sc; }
        }

        double s = best_s;
        for (int it = 0; it < newton_iters; ++it) {
            double sc = clampS_(s);
            int i = findSeg_(sc);
            double dt = sc - t_[i];

            double x   = segEval(segX_[i], dt);
            double y   = segEval(segY_[i], dt);
            double dx  = segDeriv(segX_[i], dt);
            double dy  = segDeriv(segY_[i], dt);
            double ddx = segDeriv2(segX_[i], dt);
            double ddy = segDeriv2(segY_[i], dt);

            double ex = x - double(q.x);
            double ey = y - double(q.y);

            double f1 = ex*dx + ey*dy;
            double f2 = dx*dx + dy*dy + ex*ddx + ey*ddy;
            if (std::abs(f2) < 1e-12) break;

            double step = f1 / f2;
            const double maxStep = 0.5;
            if (step >  maxStep) step =  maxStep;
            if (step < -maxStep) step = -maxStep;

            s = sc - step;
        }

        return clampS_(s);
    }
   


private:
    std::vector<double> t_;
    std::vector<SplineSegment> segX_, segY_;
    double totalLen_ = 0.0;
    bool valid_ = false;

    void clear_() {
        t_.clear();
        segX_.clear();
        segY_.clear();
        totalLen_ = 0.0;
        valid_ = false;
    }

    double clampS_(double s) const {
        if (t_.empty()) return 0.0;
        if (s <= t_.front()) return t_.front();
        if (s >= t_.back())  return t_.back();
        return s;
    }

    int findSeg_(double s) const {
        // t[i] <= s < t[i+1]
        auto it = std::upper_bound(t_.begin(), t_.end(), s);
        int idx = int(it - t_.begin()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= int(t_.size()) - 1) idx = int(t_.size()) - 2;
        return idx;
    }
};

// ============================================================
//  PERIODIC CUBIC SPLINE 1D (C^2): cyclic tridiagonal solver
// ============================================================

static void solveTridiagonal(const std::vector<double>& a,
                             const std::vector<double>& b,
                             const std::vector<double>& c,
                             const std::vector<double>& r,
                             std::vector<double>& x)
{
    const int n = (int)b.size();
    x.assign(n, 0.0);

    std::vector<double> cp(n, 0.0), dp(n, 0.0);
    double denom = b[0];
    if (std::abs(denom) < 1e-18) denom = (denom >= 0 ? 1e-18 : -1e-18);

    cp[0] = c[0] / denom;
    dp[0] = r[0] / denom;

    for (int i = 1; i < n; ++i) {
        denom = b[i] - a[i]*cp[i-1];
        if (std::abs(denom) < 1e-18) denom = (denom >= 0 ? 1e-18 : -1e-18);
        cp[i] = (i == n-1) ? 0.0 : c[i] / denom;
        dp[i] = (r[i] - a[i]*dp[i-1]) / denom;
    }

    x[n-1] = dp[n-1];
    for (int i = n-2; i >= 0; --i)
        x[i] = dp[i] - cp[i]*x[i+1];
}

static void solveCyclicTridiagonal(std::vector<double> a,
                                   std::vector<double> b,
                                   std::vector<double> c,
                                   const std::vector<double>& r,
                                   std::vector<double>& x)
{
    const int n = (int)b.size();
    x.assign(n, 0.0);
    if (n <= 2) return;

    const double alpha = a[0];
    const double beta  = c[n-1];

    a[0]   = 0.0;
    c[n-1] = 0.0;

    double gamma = -b[0];
    if (std::abs(gamma) < 1e-18) gamma = -1e-18;

    std::vector<double> bb = b;
    bb[0]   = b[0] - gamma;
    bb[n-1] = b[n-1] - (alpha*beta)/gamma;

    std::vector<double> x0;
    solveTridiagonal(a, bb, c, r, x0);

    std::vector<double> u(n, 0.0);
    u[0]   = gamma;
    u[n-1] = alpha;

    std::vector<double> z;
    solveTridiagonal(a, bb, c, u, z);

    const double fact_num = (x0[0] + (beta*x0[n-1])/gamma);
    const double fact_den = (1.0 + z[0] + (beta*z[n-1])/gamma);
    double fact = fact_num / fact_den;

    x.resize(n);
    for (int i = 0; i < n; ++i)
        x[i] = x0[i] - fact*z[i];
}

static std::vector<SplineSegment> computePeriodicCubic1D(const std::vector<double>& h,
                                                         const std::vector<double>& y)
{
    const int n = (int)y.size();
    std::vector<SplineSegment> segs;
    if (n < 3) return segs;

    std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), r(n, 0.0);
    auto idx = [&](int i)->int { return (i % n + n) % n; };

    for (int i = 0; i < n; ++i) {
        const double him1 = h[idx(i-1)];
        const double hi   = h[idx(i)];

        const int im1 = idx(i-1);
        const int ip1 = idx(i+1);

        const double slope_i   = (y[ip1] - y[i]) / hi;
        const double slope_im1 = (y[i] - y[im1]) / him1;

        a[i] = him1;
        b[i] = 2.0*(him1 + hi);
        c[i] = hi;
        r[i] = 6.0*(slope_i - slope_im1);
    }

    std::vector<double> M;
    solveCyclicTridiagonal(a, b, c, r, M);

    segs.resize(n);
    for (int i = 0; i < n; ++i) {
        const int ip1 = idx(i+1);
        const double hi = h[i];

        const double A = y[i];
        const double B = (y[ip1] - y[i]) / hi - hi*(2.0*M[i] + M[ip1])/6.0;
        const double C = M[i] / 2.0;
        const double D = (M[ip1] - M[i]) / (6.0*hi);

        segs[i] = {A, B, C, D};
    }

    return segs;
}

// ============================================================
//  CLOSED TRACK: Periodic Cubic Spline 2D (analytic)
// ============================================================

class PeriodicCubicSpline2D
{
public:
    void buildPeriodic(const std::vector<Vec2>& path)
    {
        clear_();
        if (path.size() < 3) return;

        // remove duplicates
        std::vector<Vec2> pts;
        pts.reserve(path.size());
        pts.push_back(path[0]);

        const double eps2 = 1e-12;
        for (size_t i = 1; i < path.size(); ++i) {
            double dx = path[i].x - path[i-1].x;
            double dy = path[i].y - path[i-1].y;
            if (dx*dx + dy*dy > eps2) pts.push_back(path[i]);
        }
        if (pts.size() < 3) return;

        // if last == first, drop last (loop is implicit)
        {
            double dx = pts.back().x - pts.front().x;
            double dy = pts.back().y - pts.front().y;
            if (dx*dx + dy*dy <= eps2) pts.pop_back();
        }
        if (pts.size() < 3) return;

        pts_ = pts;
        const int n = (int)pts_.size();

        h_.assign(n, 0.0);
        sKnots_.assign(n, 0.0);

        auto dist = [&](int i, int j)->double{
            double dx = (double)pts_[j].x - (double)pts_[i].x;
            double dy = (double)pts_[j].y - (double)pts_[i].y;
            return std::sqrt(dx*dx + dy*dy);
        };

        for (int i = 0; i < n; ++i) {
            int ip1 = (i+1) % n;
            h_[i] = dist(i, ip1);
            if (h_[i] < 1e-9) h_[i] = 1e-9;
        }

        for (int i = 1; i < n; ++i)
            sKnots_[i] = sKnots_[i-1] + h_[i-1];

        totalLen_ = 0.0;
        for (int i = 0; i < n; ++i) totalLen_ += h_[i];

        std::vector<double> xs(n), ys(n);
        for (int i = 0; i < n; ++i) {
            xs[i] = pts_[i].x;
            ys[i] = pts_[i].y;
        }

        segX_ = computePeriodicCubic1D(h_, xs);
        segY_ = computePeriodicCubic1D(h_, ys);

        valid_ = (!segX_.empty() && segX_.size() == segY_.size() && (int)segX_.size() == n);
    }

    bool valid() const { return valid_; }
    double totalLength() const { return totalLen_; }

    Vec2 eval(double s) const {
        if (!valid_) return Vec2(0.f, 0.f);
        double sw = wrapS_(s);
        int i = findSeg_(sw);
        double dt = sw - sKnots_[i];
        return Vec2((float)segEval(segX_[i], dt),
                    (float)segEval(segY_[i], dt));
    }

    double getX(double s) const {
        if (!valid_) return 0.0;
        double sw = wrapS_(s);
        int i = findSeg_(sw);
        double dt = sw - sKnots_[i];
        return segEval(segX_[i], dt);
    }

    double getY(double s) const {
        if (!valid_) return 0.0;
        double sw = wrapS_(s);
        int i = findSeg_(sw);
        double dt = sw - sKnots_[i];
        return segEval(segY_[i], dt);
    }

    double getYaw(double s) const {
        if (!valid_) return 0.0;
        double sw = wrapS_(s);
        int i = findSeg_(sw);
        double dt = sw - sKnots_[i];
        double dx = segDeriv(segX_[i], dt);
        double dy = segDeriv(segY_[i], dt);
        return std::atan2(dy, dx);
    }

    double getCurvature(double s) const {
        if (!valid_) return 0.0;
        double sw = wrapS_(s);
        int i = findSeg_(sw);
        double dt = sw - sKnots_[i];

        double dx  = segDeriv (segX_[i], dt);
        double dy  = segDeriv (segY_[i], dt);
        double ddx = segDeriv2(segX_[i], dt);
        double ddy = segDeriv2(segY_[i], dt);

        double num = dx*ddy - dy*ddx;
        double den = std::pow(dx*dx + dy*dy, 1.5);
        if (den < 1e-12) return 0.0;
        return num / den;
    }

    // s(x,y): periodic projection (wrap), coarse + Newton
    double projectToS(const Vec2& q,
                      double s_init,
                      double search_radius = 3.0,
                      int samples = 41,
                      int newton_iters = 20) const
    {
        if (!valid_) return 0.0;

        double best_s = wrapS_(s_init);
        double best_d2 = std::numeric_limits<double>::infinity();

        for (int k = 0; k < samples; ++k) {
            double ss = s_init - search_radius + 2.0*search_radius * double(k) / double(samples - 1);
            double sw = wrapS_(ss);
            Vec2 p = eval(sw);

            double dx = double(p.x) - double(q.x);
            double dy = double(p.y) - double(q.y);
            double d2 = dx*dx + dy*dy;

            if (d2 < best_d2) { best_d2 = d2; best_s = sw; }
        }

        double s = best_s;
        for (int it = 0; it < newton_iters; ++it) {
            double sw = wrapS_(s);
            int i = findSeg_(sw);
            double dt = sw - sKnots_[i];

            double x   = segEval(segX_[i], dt);
            double y   = segEval(segY_[i], dt);
            double dx  = segDeriv(segX_[i], dt);
            double dy  = segDeriv(segY_[i], dt);
            double ddx = segDeriv2(segX_[i], dt);
            double ddy = segDeriv2(segY_[i], dt);

            double ex = x - double(q.x);
            double ey = y - double(q.y);

            double f1 = ex*dx + ey*dy;
            double f2 = dx*dx + dy*dy + ex*ddx + ey*ddy;
            if (std::abs(f2) < 1e-12) break;

            double step = f1 / f2;
            const double maxStep = 0.5;
            if (step >  maxStep) step =  maxStep;
            if (step < -maxStep) step = -maxStep;

            s = sw - step;
        }

        return wrapS_(s);
    }

   

private:
    std::vector<Vec2> pts_;
    std::vector<double> h_;
    std::vector<double> sKnots_;
    std::vector<SplineSegment> segX_, segY_;
    double totalLen_ = 0.0;
    bool valid_ = false;

    void clear_() {
        pts_.clear();
        h_.clear();
        sKnots_.clear();
        segX_.clear();
        segY_.clear();
        totalLen_ = 0.0;
        valid_ = false;
    }

    double wrapS_(double s) const {
        if (totalLen_ <= 1e-12) return 0.0;
        s = std::fmod(s, totalLen_);
        if (s < 0.0) s += totalLen_;
        return s;
    }

    int findSeg_(double s_wrapped) const {
        // sKnots[i] <= s < sKnots[i+1], last: [sKnots[n-1], totalLen)
        auto it = std::upper_bound(sKnots_.begin(), sKnots_.end(), s_wrapped);
        int idx = int(it - sKnots_.begin()) - 1;
        if (idx < 0) idx = 0;
        if (idx >= int(sKnots_.size())) idx = int(sKnots_.size()) - 1;
        return idx;
    }
};

// ============================================================
//  WRAPPER: one public API, chooses backend:
//   - periodic (closed) -> PeriodicCubicSpline2D
//   - open             -> OpenAkimaSpline2D
// ============================================================

class TrackSpline2D
{
public:
    void build(const std::vector<Vec2>& path, bool closed_loop)
    {
        closed_ = closed_loop;

        if (closed_) {
            cubic_.buildPeriodic(path);
            akima_ = OpenAkimaSpline2D(); // reset
        } else {
            akima_.buildOpen(path);
            cubic_ = PeriodicCubicSpline2D(); // reset
        }
    }

    bool valid() const {
        return closed_ ? cubic_.valid() : akima_.valid();
    }

    bool isClosed() const { return closed_; }

    double totalLength() const {
        return closed_ ? cubic_.totalLength() : akima_.totalLength();
    }

    Vec2 eval(double s) const {
        return closed_ ? cubic_.eval(s) : akima_.eval(s);
    }

    double getX(double s) const {
        return closed_ ? cubic_.getX(s) : akima_.getX(s);
    }

    double getY(double s) const {
        return closed_ ? cubic_.getY(s) : akima_.getY(s);
    }

    double getYaw(double s) const {
        return closed_ ? cubic_.getYaw(s) : akima_.getYaw(s);
    }

    double getCurvature(double s) const {
        return closed_ ? cubic_.getCurvature(s) : akima_.getCurvature(s);
    }

    double projectToS(const Vec2& q,
                      double s_init,
                      double search_radius = 3.0,
                      int samples = 41,
                      int newton_iters = 20) const
    {
        return closed_
            ? cubic_.projectToS(q, s_init, search_radius, samples, newton_iters)
            : akima_.projectToS(q, s_init, search_radius, samples, newton_iters);
    }


    // ------------------------------------------------------------
    // NEW: signed normal distance from spline to point q
    // + : q on the left of tangent (in direction of increasing s)
    // - : q on the right
    // Optionally returns s_proj and projected point p_proj.
    // ------------------------------------------------------------
    double signedNormalDistance(const Vec2& q,
                                double s_init,
                                double search_radius = 3.0,
                                int samples = 41,
                                int newton_iters = 20,
                                double* s_proj_out = nullptr,
                                Vec2* p_proj_out = nullptr) const
    {
        if (!valid()) {
            if (s_proj_out) *s_proj_out = 0.0;
            if (p_proj_out) *p_proj_out = Vec2(0.f, 0.f);
            return 0.0;
        }

        const double s_proj = projectToS(q, s_init, search_radius, samples, newton_iters);
        const Vec2 p = eval(s_proj);

        const double vx = double(q.x) - double(p.x);
        const double vy = double(q.y) - double(p.y);

        const double dist = std::sqrt(vx*vx + vy*vy);

        const double yaw = getYaw(s_proj);
        const double tx = std::cos(yaw);
        const double ty = std::sin(yaw);

        const double c = cross2(tx, ty, vx, vy);
        const double signed_dist = (c >= 0.0) ? dist : -dist;

        if (s_proj_out) *s_proj_out = s_proj;
        if (p_proj_out) *p_proj_out = p;

        return signed_dist;
    }

    // Convenience: unsigned distance (absolute value)
    double normalDistance(const Vec2& q,
                          double s_init,
                          double search_radius = 3.0,
                          int samples = 41,
                          int newton_iters = 20,
                          double* s_proj_out = nullptr,
                          Vec2* p_proj_out = nullptr) const
    {
        const double d = signedNormalDistance(q, s_init, search_radius, samples, newton_iters, s_proj_out, p_proj_out);
        return std::abs(d);
    }

    
    double projectToSpline(const double x, const double y) const
    {
        return projectToSpline(Vec2((float)x, (float)y));
    }

    double projectToSpline(const Vec2& q) const
    {
        if (!valid()) return 0.0;

        const double L = totalLength();
        if (L <= 1e-12) return 0.0;

        // heurystyka: ~co 0.5 m, z clampem
        const double coarse_step = 0.5;
        int coarse_samples = (int)std::ceil(L / std::max(1e-6, coarse_step));
        coarse_samples = std::max(25, std::min(coarse_samples, 2000));

        double best_s  = 0.0;
        double best_d2 = std::numeric_limits<double>::infinity();

        if (closed_) {
            // closed: próbkuj [0, L) bez duplikatu końca
            for (int i = 0; i < coarse_samples; ++i) {
                const double s = (L * double(i)) / double(coarse_samples); // [0, L)
                const Vec2 p = eval(s);

                const double dx = double(p.x) - double(q.x);
                const double dy = double(p.y) - double(q.y);
                const double d2 = dx*dx + dy*dy;

                if (d2 < best_d2) { best_d2 = d2; best_s = s; }
            }
        } else {
            // open: próbkuj [0, L] z endpointem
            for (int i = 0; i < coarse_samples; ++i) {
                const double s = (L * double(i)) / double(coarse_samples - 1); // [0, L]
                const Vec2 p = eval(s);

                const double dx = double(p.x) - double(q.x);
                const double dy = double(p.y) - double(q.y);
                const double d2 = dx*dx + dy*dy;

                if (d2 < best_d2) { best_d2 = d2; best_s = s; }
            }
        }

        // refinement na Twojej istniejącej metodzie (domyślne argumenty)
        return projectToS(q, best_s);
    }

private:
    bool closed_ = false;
    PeriodicCubicSpline2D cubic_;
    OpenAkimaSpline2D akima_;
};

} // namespace v2_control
