#pragma once

#include <cstdint>

namespace Erosion {

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Filter {
    float strength = 0.f;
    float scale = 0.f;
    std::uint32_t octaves = 4;
    std::uint32_t seed = 1234;

    // p must be a unit sphere position; pixel_angle is the pixel size in radians.
    float evaluate(Vec3 p, Vec3 gradient, float height, float pixel_angle) const;
};

} // namespace Erosion
