#include <unblending.h>
#include "pixel_solver.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <limits>
#include <new>

namespace {
using namespace unblending;

gpu::Backend select_backend(int32_t backend)
{
    switch (backend)
    {
    case UNBLENDING_BACKEND_AUTO: return gpu::Backend::automatic;
    case UNBLENDING_BACKEND_CPU: return gpu::Backend::cpu;
    case UNBLENDING_BACKEND_CUDA: return gpu::Backend::cuda;
    default: throw std::invalid_argument("Unknown backend");
    }
}

std::vector<LayerInfo> make_layers(const unblending_layer* layers, size_t count)
{
    static constexpr BlendMode modes[] = {
        BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
        BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
        BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference,
        BlendMode::Exclusion, BlendMode::LinearDodge
    };

    std::vector<LayerInfo> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        const auto& layer = layers[i];
        if (layer.blend_mode < UNBLENDING_NORMAL || layer.blend_mode > UNBLENDING_LINEAR_DODGE)
            throw std::invalid_argument("Unknown blend mode");
        const Vec3 mean = Eigen::Map<const Vec3>(layer.mean);
        const Mat3 precision = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(layer.precision);
        if (!mean.allFinite() || !precision.allFinite() || !precision.isApprox(precision.transpose(), 1e-12) ||
            Eigen::LLT<Mat3>(precision).info() != Eigen::Success)
            throw std::invalid_argument("Invalid Gaussian color model");
        result.push_back({CompOp(layer.comp_x, layer.comp_y, layer.comp_z), modes[layer.blend_mode],
                          std::make_shared<GaussianColorModel>(mean, precision)});
    }
    return result;
}

int32_t solve(const unblending_rgb_view* image, const unblending_layer* layers, size_t layer_count,
              const unblending_options* options, const RgbaLayersView* first,
              double* output, size_t output_capacity)
{
    try
    {
        constexpr auto max_int = std::numeric_limits<int>::max();
        if (!image || !layers || !layer_count || layer_count > size_t(max_int) / 4 ||
            image->width <= 0 || image->height <= 0 || image->width > max_int / image->height)
            return UNBLENDING_INVALID_ARGUMENT;

        const size_t pixels = size_t(image->width) * image->height;
        if (pixels > std::vector<double>().max_size() / 4 / layer_count)
            return UNBLENDING_INVALID_ARGUMENT;
        const size_t required = 4 * pixels * layer_count;
        if (output_capacity < required) return UNBLENDING_BUFFER_TOO_SMALL;
        if (!output) return UNBLENDING_INVALID_ARGUMENT;

        const unblending_options defaults = UNBLENDING_OPTIONS_INIT;
        if (!options) options = &defaults;
        if (options->struct_size < sizeof(unblending_options) ||
            (options->opaque_background != 0 && options->opaque_background != 1) ||
            (options->smooth_background != 0 && options->smooth_background != 1) ||
            options->target_concurrency < 0)
            return UNBLENDING_INVALID_ARGUMENT;

        const auto backend = select_backend(options->backend);
        const RgbImageView rgb(image->data, image->size, image->width, image->height, image->row_stride);
        const auto specs = make_layers(layers, layer_count);
        const bool opaque = options->opaque_background != 0;
        const auto result = first
            ? detail::refine_pixels(rgb, *first, specs, opaque, options->smooth_background != 0,
                                    options->target_concurrency, backend)
            : detail::unmix_pixels(rgb, specs, opaque, options->target_concurrency, backend);
        std::copy(result.begin(), result.end(), output);
        return UNBLENDING_OK;
    }
    catch (const std::bad_alloc&) { return UNBLENDING_OUT_OF_MEMORY; }
    catch (const gpu::ExecutionError&) { return UNBLENDING_CUDA_ERROR; }
    catch (const std::invalid_argument&) { return UNBLENDING_INVALID_ARGUMENT; }
    catch (const std::length_error&) { return UNBLENDING_INVALID_ARGUMENT; }
    catch (...) { return UNBLENDING_SOLVER_ERROR; }
}
}

uint32_t unblending_abi_version(void) { return UNBLENDING_ABI_VERSION; }

int32_t unblending_unmix(const unblending_rgb_view* image, const unblending_layer* layers,
                         size_t layer_count, const unblending_options* options,
                         double* output, size_t output_capacity)
{
    return solve(image, layers, layer_count, options, nullptr, output, output_capacity);
}

int32_t unblending_refine(const unblending_rgb_view* image, const unblending_layer* layers,
                          size_t layer_count, const double* first_pass, size_t first_pass_size,
                          const unblending_options* options, double* output, size_t output_capacity)
{
    const unblending::RgbaLayersView first{first_pass, first_pass_size};
    return solve(image, layers, layer_count, options, &first, output, output_capacity);
}
