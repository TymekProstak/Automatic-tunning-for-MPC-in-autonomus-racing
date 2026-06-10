#pragma once
#include <cmath>

namespace v2_control
{
struct Vec2 {
    double x;
    double y;

    Vec2() : x(0.0f), y(0.0f) {}
    Vec2(double xx, double yy) : x(xx), y(yy) {}

    inline Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    inline Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    inline Vec2 operator*(double scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    inline Vec2& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    inline double length() const {
        return std::sqrt(x * x + y * y);
    }
};
}