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
namespace firstpass {
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

    constexpr double blend_function_internal_epsilon = 1e-05;

    HD inline double blend_grad_s(double s, double d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return 1.0;
            case BlendMode::Multiply:
                return d;
            case BlendMode::Screen:
                return 1.0 - d;
            case BlendMode::Overlay:
                return (d <= 0.5) ? 2.0 * d : 2.0 * (1.0 - d);
            case BlendMode::Darken:
                return (s < d) ? 1.0 : 0.0;
            case BlendMode::Lighten:
                return (s < d) ? 0.0 : 1.0;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0 : ((1.0 - s < blend_function_internal_epsilon) ? 0.0 : ((1.0 < d / (1.0 - s)) ? 0.0 : d / ((1.0 - s) * (1.0 - s))));
            case BlendMode::ColorBurn:
                return (1.0 - d < blend_function_internal_epsilon) ? 0.0 : ((s < blend_function_internal_epsilon) ? 0.0 : - (1.0 < (1.0 - d) / s ? 0.0 : - (1.0 - d) / (s * s)));
            case BlendMode::HardLight:
                return (s <= 0.5) ? 2.0 * d : 2.0 * (1.0 - d);
            case BlendMode::SoftLight:
                return (s <= 0.5) ? 2.0 * d * (1.0 - d) : 2.0 * (((d <= 0.25) ? ((16.0 * d - 12.0) * d + 4.0) * d : sqrt(d)) - d);
            case BlendMode::Difference:
                return (s < d) ? - 1.0 : 1.0;
            case BlendMode::Exclusion:
                return 1.0 - 2.0 * d;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0) ? 0.0 : 1.0;
            default:
                assert(false);
                return 0.0;
        }
    }

    HD inline double blend_grad_d(double s, double d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return 0.0;
            case BlendMode::Multiply:
                return s;
            case BlendMode::Screen:
                return 1.0 - s;
            case BlendMode::Overlay:
                return (d <= 0.5) ? 2.0 * s : 2.0 * (1.0 - s);
            case BlendMode::Darken:
                return (s < d) ? 0.0 : 1.0;
            case BlendMode::Lighten:
                return (s < d) ? 1.0 : 0.0;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0 : ((1.0 - s < blend_function_internal_epsilon) ? 0.0 : ((1.0 < d / (1.0 - s)) ? 0.0 : 1.0 / (1.0 - s)));
            case BlendMode::ColorBurn:
                return (1.0 - d < blend_function_internal_epsilon) ? 0.0 : ((s < blend_function_internal_epsilon) ? 0.0 : - (1.0 < (1.0 - d) / s ? 0.0 : - 1.0 / s));
            case BlendMode::HardLight:
                return (s <= 0.5) ? 2.0 * s : 2.0 * (1.0 - s);
            case BlendMode::SoftLight:
                return (s <= 0.5) ? 2.0 * s + 2.0 * d - 4.0 * s * d : (1.0 + (2.0 * s - 1.0) * ((d <= 0.25) ? (48.0 * d * d - 24.0 * d + 4.0) : (1.0 / (2.0 * sqrt(d)))) - (2.0 * s - 1.0));
            case BlendMode::Difference:
                return (s < d) ? 1.0 : - 1.0;
            case BlendMode::Exclusion:
                return 1.0 - 2.0 * s;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0) ? 0.0 : 1.0;
            default:
                assert(false);
                return 0.0;
        }
    }

    HD inline double blend(double s, double d, BlendMode mode, bool crop = false)
    {
        switch (mode)
        {
            case BlendMode::Normal:
                return s;
            case BlendMode::Multiply:
                return s * d;
            case BlendMode::Screen:
                return 1.0 - (1.0 - s) * (1.0 - d);
            case BlendMode::Overlay:
                return (d <= 0.5) ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
            case BlendMode::Darken:
                return (s < d) ? s : d;
            case BlendMode::Lighten:
                return (s < d) ? d : s;
            case BlendMode::ColorDodge:
                return (d < blend_function_internal_epsilon) ? 0.0 : ((1.0 - s < blend_function_internal_epsilon) ? 1.0 : fmin(1.0, d / (1.0 - s)));
            case BlendMode::ColorBurn:
                return (1.0 - d < blend_function_internal_epsilon) ? 1.0 : ((s < blend_function_internal_epsilon) ? 0.0 : 1.0 - fmin(1.0, (1.0 - d) / s));
            case BlendMode::HardLight:
                return (s <= 0.5) ? 2.0 * s * d : 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
            case BlendMode::SoftLight:
                return (s <= 0.5) ? d - (1.0 - 2.0 * s) * d * (1.0 - d) : (d + (2.0 * s - 1.0) * (((d <= 0.25) ? ((16.0 * d - 12.0) * d + 4.0) * d : sqrt(d)) - d));
            case BlendMode::Difference:
                return (s < d) ? d - s : s - d;
            case BlendMode::Exclusion:
                return s + d - 2.0 * s * d;
            case BlendMode::LinearDodge:
                return (crop && s + d > 1.0) ? 1.0 : s + d;
            default:
                assert(false);
                return 0.0;
        }
    }

}
