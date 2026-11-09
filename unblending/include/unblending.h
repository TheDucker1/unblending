#ifndef UNBLENDING_C_API_H
#define UNBLENDING_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
# if defined(UNBLENDING_SDK_EXPORTS)
#  define UNBLENDING_API __declspec(dllexport)
# else
#  define UNBLENDING_API __declspec(dllimport)
# endif
#else
# define UNBLENDING_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define UNBLENDING_ABI_VERSION 1u

enum {
    UNBLENDING_OK = 0,
    UNBLENDING_INVALID_ARGUMENT = 1,
    UNBLENDING_BUFFER_TOO_SMALL = 2,
    UNBLENDING_CUDA_ERROR = 3,
    UNBLENDING_OUT_OF_MEMORY = 4,
    UNBLENDING_SOLVER_ERROR = 5
};

enum {
    UNBLENDING_BACKEND_AUTO = 0,
    UNBLENDING_BACKEND_CPU = 1,
    UNBLENDING_BACKEND_CUDA = 2
};

enum {
    UNBLENDING_NORMAL = 0,
    UNBLENDING_MULTIPLY = 1,
    UNBLENDING_SCREEN = 2,
    UNBLENDING_OVERLAY = 3,
    UNBLENDING_DARKEN = 4,
    UNBLENDING_LIGHTEN = 5,
    UNBLENDING_COLOR_DODGE = 6,
    UNBLENDING_COLOR_BURN = 7,
    UNBLENDING_HARD_LIGHT = 8,
    UNBLENDING_SOFT_LIGHT = 9,
    UNBLENDING_DIFFERENCE = 10,
    UNBLENDING_EXCLUSION = 11,
    UNBLENDING_LINEAR_DODGE = 12
};

/* RGB triples in [0,1]. Counts and strides are in doubles.
 * A zero stride selects tightly packed rows (3 * width). */
typedef struct unblending_rgb_view {
    const double* data;
    size_t size;
    int32_t width;
    int32_t height;
    size_t row_stride;
} unblending_rgb_view;

/* Bottom-to-top layer order. Source-over is (1,1,1), Plus is (2,1,1).
 * Mean uses the normalized RGB scale. Precision is a positive-definite inverse
 * covariance matrix in row-major order. */
typedef struct unblending_layer {
    int32_t comp_x, comp_y, comp_z;
    int32_t blend_mode;
    double mean[3];
    double precision[9];
} unblending_layer;

typedef struct unblending_options {
    uint32_t struct_size;
    int32_t backend;
    int32_t opaque_background;
    int32_t smooth_background;
    int32_t target_concurrency;
} unblending_options;

#define UNBLENDING_OPTIONS_INIT \
    { (uint32_t)sizeof(unblending_options), UNBLENDING_BACKEND_AUTO, 1, 0, 0 }

UNBLENDING_API uint32_t unblending_abi_version(void);

/* Outputs are caller-owned, packed [layer][row][column][RGBA] doubles
 * with straight alpha. Supply capacity for 4 * width * height * layer_count
 * doubles. Inputs are borrowed for the call. Output is written on success.
 * NULL options select UNBLENDING_OPTIONS_INIT. CUDA mode requires GPU
 * execution. A concurrency of zero selects the default CPU worker count. */
UNBLENDING_API int32_t unblending_unmix(
    const unblending_rgb_view* image, const unblending_layer* layers, size_t layer_count,
    const unblending_options* options, double* output, size_t output_capacity);

/* The first pass must contain exactly 4 * width * height * layer_count doubles.
 * It may alias output. Background smoothing applies to this function. */
UNBLENDING_API int32_t unblending_refine(
    const unblending_rgb_view* image, const unblending_layer* layers, size_t layer_count,
    const double* first_pass, size_t first_pass_size, const unblending_options* options,
    double* output, size_t output_capacity);

#ifdef __cplusplus
}
#endif
#endif
