// Advanced Terrain Erosion Filter and Phacelle Noise:
// Copyright (c) 2025 Rune Skovbo Johansen.
// C# adaptation copyright (c) 2026 Luke Mitchell.
// Spherical C++ adaptation of ErosionFilter and PhacelleNoise from:
// https://github.com/lpmitchell/AdvancedTerrainErosion/blob/main/package/Runtime/LPMGames.Terrain.Erosion/AdvancedTerrainErosion.cs
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Erosion.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace Erosion {
namespace {

    constexpr float PI = 3.14159265358979323846f;

    Vec3 operator+(const Vec3& a, const Vec3& b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    Vec3 operator-(const Vec3& a, const Vec3& b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    Vec3 operator*(const Vec3& v, float scale) {
        return { v.x * scale, v.y * scale, v.z * scale };
    }

    Vec3 operator*(float scale, const Vec3& v) {
        return v * scale;
    }

    Vec3 operator/(const Vec3& v, float scale) {
        return { v.x / scale, v.y / scale, v.z / scale };
    }

    float dot_prod(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    float norm(const Vec3& v) {
        return std::sqrt(dot_prod(v, v));
    }

    Vec3 cross_prod(const Vec3& a, const Vec3& b) {
        return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

    float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    inline std::uint32_t hash(std::uint32_t value) {
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        return value ^ (value >> 16);
    }

    inline Vec3 cell_offset(const std::array<int, 3>& cell, std::uint32_t seed) {
        std::uint32_t state = hash(std::uint32_t(cell[0]) ^ seed);
        state = hash(state ^ std::uint32_t(cell[1]));
        state = hash(state ^ std::uint32_t(cell[2]));
        std::array<float, 3> offset{};
        for (int axis = 0; axis < 3; ++axis) {
            state = hash(state + std::uint32_t(axis) + 1u);
            offset[axis] = float(state >> 8) / 16777216.f - 0.5f;
        }
        return { offset[0], offset[1], offset[2] };
    }

    // A 3D cell field restricted to the sphere. side points across the
    // downhill direction in the tangent plane; no latitude basis is needed.
    inline std::array<float, 2> phacelle(const Vec3& p, const Vec3& side, std::uint32_t seed) {
        const std::array<int, 3> cell{
            int(std::floor(p.x)), int(std::floor(p.y)), int(std::floor(p.z))
        };
        const Vec3 fraction = p - Vec3{ float(cell[0]), float(cell[1]), float(cell[2]) };
        std::array<float, 2> phase{};
        float weights = 0.f;
        for (int z = -1; z <= 2; ++z) {
            for (int y = -1; y <= 2; ++y) {
                for (int x = -1; x <= 2; ++x) {
                    const Vec3 delta = fraction - Vec3{ float(x), float(y), float(z) } -
                                       cell_offset({ cell[0] + x, cell[1] + y, cell[2] + z }, seed);
                    const float distance2 = dot_prod(delta, delta);
                    if (distance2 >= 2.25f) {
                        continue;
                    }
                    // Compact support makes cell-boundary transitions continuous.
                    const float weight =
                        std::max(0.f, std::exp(-2.f * distance2) - std::exp(-4.5f));
                    const float angle = dot_prod(delta, side) + 0.5f * PI;
                    phase[0] += weight * std::cos(angle);
                    phase[1] += weight * std::sin(angle);
                    weights += weight;
                }
            }
        }
        phase[0] /= std::max(weights, 1.e-6f);
        phase[1] /= std::max(weights, 1.e-6f);
        const float divisor = std::max(0.5f, std::sqrt(phase[0] * phase[0] + phase[1] * phase[1]));
        return { phase[0] / divisor, phase[1] / divisor };
    }

    inline float mask(float slope, float rounding) {
        const float smooth =
            slope >= rounding ? slope - 0.5f * rounding : 0.5f * slope * slope / rounding;
        const float remainder = 1.f - std::clamp(smooth, 0.f, 1.f);
        return 1.f - remainder * remainder;
    }

} // namespace

float Filter::evaluate(Vec3 p, Vec3 gradient, float height, float pixel_angle) const {
    if (strength <= 0.f || scale <= 0.f) {
        return height;
    }

    const Vec3 input_slope = 0.5f * (gradient - p * dot_prod(gradient, p));
    const float slope_length = norm(input_slope);
    const float initial_target = std::clamp(height / 0.6f, -1.f, 1.f);

    Vec3 gully_slope = input_slope / std::max(slope_length, 1.e-10f) * 0.7f;
    const float input_rounding =
        lerp(0.f, 0.1f, std::clamp(initial_target + 0.5f, 0.f, 1.f)) * 0.1f;
    float combined_mask = mask(slope_length * 1.25f, input_rounding * 1.25f);
    float target = initial_target;
    float amplitude = strength * scale;
    constexpr float cell_scale = 0.7f;
    float frequency = 1.f / (scale * cell_scale);
    float rounding_multiplier = 1.f;
    float delta = 0.f;
    float magnitude = 0.f;
    for (std::uint32_t octave = 0; octave < octaves; ++octave) {
        const float resolved =
            std::clamp((1.f / (frequency * cell_scale * pixel_angle) - 2.f) / 2.f, 0.f, 1.f);
        if (resolved <= 0.f) {
            break;
        }
        const Vec3 direction = gully_slope / std::max(norm(gully_slope), 1.e-6f);
        const Vec3 side = cross_prod(p, direction) * (cell_scale * 2.f * PI);
        const auto wave = phacelle(p * frequency, side, seed);
        const Vec3 slope = -frequency * side;
        const float sign = wave[1] > 0.f ? 1.f : (wave[1] < 0.f ? -1.f : 0.f);
        gully_slope = gully_slope + sign * slope * amplitude * 0.5f * resolved;
        const float gullies = lerp(target, wave[0] * 0.5f, combined_mask);
        delta += gullies * amplitude * resolved;
        magnitude += amplitude * resolved;
        target = gullies;
        const float rounding =
            lerp(0.f, 0.1f, std::clamp(wave[0] + 0.5f, 0.f, 1.f)) * rounding_multiplier;
        combined_mask = (1.f - std::pow(1.f - std::clamp(combined_mask, 0.f, 1.f), 1.5f)) *
                        mask(std::abs(wave[1]) * 1.25f, rounding * 1.25f);
        amplitude *= 0.5f;
        frequency *= 2.f;
        rounding_multiplier *= 2.f;
    }
    return height + 2.f * (delta - 0.65f * magnitude);
}

} // namespace Erosion
