// Single-precision scalar blend functions for CUDA.
// Scalar blend functions adapted from unblending/include/unblending/blend_mode.hpp.
// Copyright (c) 2018 AIST; same dual licensing as the parent repository.
#pragma once
#include <cmath>
#include <cassert>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
namespace firstpass32 {
    enum class BlendMode
    {
        Normal,
        Multiply,
        Screen,
        Overlay,
        Darken,
        Lighten,
        ColorDodge,
        ColorBurn,
        HardLight,
        SoftLight,
        Difference,
        Exclusion,
        LinearDodge,
    };

    constexpr float blend_function_internal_epsilon = 1e-05f;

    HD inline float blend_grad_s(float s, float d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return 1.0f;
            case BlendMode::Multiply:
                return d;
            case BlendMode::Screen:
                return 1.0f - d;
            case BlendMode::Overlay:
                return (d <= 0.5f) ? 2.0f * d : 2.0f * (1.0f - d);
            case BlendMode::Darken:
                return (s < d) ? 1.0f : 0.0f;
            case BlendMode::Lighten:
                return (s < d) ? 0.0f : 1.0f;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0f : ((1.0f - s < blend_function_internal_epsilon) ? 0.0f : ((1.0f < d / (1.0f - s)) ? 0.0f : d / ((1.0f - s) * (1.0f - s))));
            case BlendMode::ColorBurn:
                return (1.0f - d < blend_function_internal_epsilon) ? 0.0f : ((s < blend_function_internal_epsilon) ? 0.0f : - (1.0f < (1.0f - d) / s ? 0.0f : - (1.0f - d) / (s * s)));
            case BlendMode::HardLight:
                return (s <= 0.5f) ? 2.0f * d : 2.0f * (1.0f - d);
            case BlendMode::SoftLight:
                return (s <= 0.5f) ? 2.0f * d * (1.0f - d) : 2.0f * (((d <= 0.25f) ? ((16.0f * d - 12.0f) * d + 4.0f) * d : sqrt(d)) - d);
            case BlendMode::Difference:
                return (s < d) ? - 1.0f : 1.0f;
            case BlendMode::Exclusion:
                return 1.0f - 2.0f * d;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0f) ? 0.0f : 1.0f;
            default:
                assert(false);
                return 0.0f;
        }
    }

    HD inline float blend_grad_d(float s, float d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return 0.0f;
            case BlendMode::Multiply:
                return s;
            case BlendMode::Screen:
                return 1.0f - s;
            case BlendMode::Overlay:
                return (d <= 0.5f) ? 2.0f * s : 2.0f * (1.0f - s);
            case BlendMode::Darken:
                return (s < d) ? 0.0f : 1.0f;
            case BlendMode::Lighten:
                return (s < d) ? 1.0f : 0.0f;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0f : ((1.0f - s < blend_function_internal_epsilon) ? 0.0f : ((1.0f < d / (1.0f - s)) ? 0.0f : 1.0f / (1.0f - s)));
            case BlendMode::ColorBurn:
                return (1.0f - d < blend_function_internal_epsilon) ? 0.0f : ((s < blend_function_internal_epsilon) ? 0.0f : - (1.0f < (1.0f - d) / s ? 0.0f : - 1.0f / s));
            case BlendMode::HardLight:
                return (s <= 0.5f) ? 2.0f * s : 2.0f * (1.0f - s);
            case BlendMode::SoftLight:
                return (s <= 0.5f) ? 2.0f * s + 2.0f * d - 4.0f * s * d : (1.0f + (2.0f * s - 1.0f) * ((d <= 0.25f) ? (48.0f * d * d - 24.0f * d + 4.0f) : (1.0f / (2.0f * sqrt(d)))) - (2.0f * s - 1.0f));
            case BlendMode::Difference:
                return (s < d) ? 1.0f : - 1.0f;
            case BlendMode::Exclusion:
                return 1.0f - 2.0f * s;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0f) ? 0.0f : 1.0f;
            default:
                assert(false);
                return 0.0f;
        }
    }

    HD inline float blend(float s, float d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return s;
            case BlendMode::Multiply:
                return s * d;
            case BlendMode::Screen:
                return 1.0f - (1.0f - s) * (1.0f - d);
            case BlendMode::Overlay:
                return (d <= 0.5f) ? 2.0f * s * d : 1.0f - 2.0f * (1.0f - s) * (1.0f - d);
            case BlendMode::Darken:
                return (s < d) ? s : d;
            case BlendMode::Lighten:
                return (s < d) ? d : s;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0f : ((1.0f - s < blend_function_internal_epsilon) ? 1.0f : fmin(1.0f, d / (1.0f - s)));
            case BlendMode::ColorBurn:
                return (1.0f - d < blend_function_internal_epsilon) ? 1.0f : ((s < blend_function_internal_epsilon) ? 0.0f : 1.0f - fmin(1.0f, (1.0f - d) / s));
            case BlendMode::HardLight:
                return (s <= 0.5f) ? 2.0f * s * d : 1.0f - 2.0f * (1.0f - s) * (1.0f - d);
            case BlendMode::SoftLight:
                return (s <= 0.5f) ? d - (1.0f - 2.0f * s) * d * (1.0f - d) : (d + (2.0f * s - 1.0f) * (((d <= 0.25f) ? ((16.0f * d - 12.0f) * d + 4.0f) * d : sqrt(d)) - d));
            case BlendMode::Difference:
                return (s < d) ? d - s : s - d;
            case BlendMode::Exclusion:
                return s + d - 2.0f * s * d;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0f) ? 1.0f : s + d;
            default:
                assert(false);
                return 0.0f;
        }
    }

}
