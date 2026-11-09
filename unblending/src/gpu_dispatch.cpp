#include "gpu_dispatch.hpp"
#include <unblending/unblending.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace unblending {
namespace gpu {

#ifdef UNBLENDING_HAS_CUDA
namespace {
std::vector<Layer> describe(const std::vector<LayerInfo>& layers)
{
    std::vector<Layer> result;
    result.reserve(layers.size());
    for (const auto& layer : layers)
    {
        const auto* model = dynamic_cast<const GaussianColorModel*>(layer.color_model.get());
        if (!model)
            throw std::invalid_argument("CUDA requires a GaussianColorModel for each layer");
        const auto& precision = model->get_sigma_inv();
        if (!model->get_mu().allFinite() || !precision.allFinite() ||
            !precision.isApprox(precision.transpose(), 1e-12))
            throw std::invalid_argument("CUDA requires a finite mean and symmetric precision");

        const int mode = static_cast<int>(layer.blend_mode);
        if (mode < static_cast<int>(BlendMode::Normal) || mode > static_cast<int>(BlendMode::LinearDodge))
            throw std::invalid_argument("Unsupported CUDA blend mode");

        Layer data{};
        data.x = layer.comp_op.X;
        data.y = layer.comp_op.Y;
        data.z = layer.comp_op.Z;
        data.mode = mode;
        for (int r = 0; r < 3; ++r)
        {
            data.mean[r] = model->get_mu()(r);
            for (int s = 0; s < 3; ++s)
                data.precision[3 * r + s] = precision(r, s);
        }
        result.push_back(data);
    }
    return result;
}
} // namespace
#endif

namespace {
template<class ReadRgb, class ReadRgba>
bool try_solve_planes(int w, int h, const std::vector<LayerInfo>& layers, bool refinement,
                      bool valid_layers, Options options, ReadRgb read_rgb, ReadRgba read_rgba,
                      std::vector<double>& output)
{
    const char* backend = std::getenv("UNBLENDING_BACKEND");
    const bool required = backend && std::strcmp(backend, "cuda") == 0;
    if (backend && std::strcmp(backend, "cpu") == 0) return false;
    if (backend && *backend && !required && std::strcmp(backend, "auto") != 0)
        throw std::invalid_argument("UNBLENDING_BACKEND must be auto, cpu, or cuda");

#ifndef UNBLENDING_HAS_CUDA
    if (required) throw std::runtime_error("Rebuild with UNBLENDING_ENABLE_CUDA=ON");
    return false;
#else
    try
    {
        // The streaming filter indexes up to sixteen planes with signed ints.
        constexpr size_t max_pixels = (std::numeric_limits<int>::max() - 255) / 16;
        if (w <= 0 || h <= 0 || size_t(w) > max_pixels / size_t(h) ||
            layers.empty() || layers.size() > 4096 || !valid_layers)
            throw std::invalid_argument("Image or layer dimensions are outside the CUDA solver domain");
        if (!options.supported_build)
            throw std::invalid_argument("This solver build uses CPU-specific initialization or tolerances");

        auto model = describe(layers);
        require_device();
        const size_t n = size_t(w) * h, l = layers.size();
        std::vector<double> rgb(3 * n);
        std::vector<double> initial(refinement ? 4 * l * n : 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = size_t(y) * w + x;
                const auto pixel = read_rgb(x, y);
                for (int r = 0; r < 3; ++r) rgb[size_t(r) * n + i] = pixel(r);
            }
        if (refinement)
            for (size_t layer = 0; layer < l; ++layer)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                    {
                        const size_t i = size_t(y) * w + x;
                        const auto pixel = read_rgba(layer, x, y);
                        initial[layer * n + i] = pixel(3);
                        for (int r = 0; r < 3; ++r)
                            initial[(l + 3 * layer + r) * n + i] = pixel(r);
                    }

        output = solve_cuda(w, h, model, rgb, refinement ? &initial : nullptr, options);
        return true;
    }
    catch (const std::logic_error& error)
    {
        if (required) throw std::runtime_error(std::string("CUDA unblending: ") + error.what());
    }
    catch (const std::runtime_error& error)
    {
        if (required) throw std::runtime_error(std::string("CUDA unblending: ") + error.what());
    }
    return false;
#endif
}

} // namespace

bool try_solve(const ColorImage& image, const std::vector<LayerInfo>& layers,
               const std::vector<ColorImage>* first, Options options,
               std::vector<ColorImage>& output)
{
    const int w = image.width(), h = image.height();
    const bool valid_layers = !first || (first->size() == layers.size() &&
        std::all_of(first->begin(), first->end(), [=](const ColorImage& layer) {
            return layer.width() == w && layer.height() == h;
        }));
    const auto read_rgb = [&](int x, int y) { return image.get_rgb(x, y); };
    const auto read_rgba = [&](std::size_t l, int x, int y) { return (*first)[l].get_rgba(x, y); };
    std::vector<double> solved;
    if (!try_solve_planes(w, h, layers, first != nullptr, valid_layers, options,
                          read_rgb, read_rgba, solved))
        return false;

    const std::size_t n = std::size_t(w) * h, layer_count = layers.size();
    std::vector<ColorImage> result;
    result.reserve(layer_count);
    for (std::size_t l = 0; l < layer_count; ++l)
    {
        result.emplace_back(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const std::size_t i = std::size_t(y) * w + x;
                const std::size_t color = (layer_count + 3 * l) * n + i;
                result.back().set_rgba(x, y, Vec3(solved[color], solved[color + n], solved[color + 2 * n]),
                                       solved[l * n + i]);
            }
    }
    output = std::move(result);
    return true;
}

bool try_solve_pixels(RgbImageView image, const std::vector<LayerInfo>& layers,
                      const RgbaLayersView* first, Options options,
                      std::vector<double>& output)
{
    const std::size_t n = std::size_t(image.width) * image.height;
    const auto read_rgb = [&](int x, int y) {
        const double* rgb = image.data + y * image.row_stride + 3 * std::size_t(x);
        return Vec3(rgb[0], rgb[1], rgb[2]);
    };
    const auto read_rgba = [&](std::size_t l, int x, int y) {
        const double* rgba = first->data + 4 * (l * n + std::size_t(y) * image.width + x);
        return Vec4(rgba[0], rgba[1], rgba[2], rgba[3]);
    };
    std::vector<double> solved;
    if (!try_solve_planes(image.width, image.height, layers, first != nullptr, true,
                          options, read_rgb, read_rgba, solved))
        return false;

    const std::size_t layer_count = layers.size();
    output.resize(solved.size());
    for (std::size_t l = 0; l < layer_count; ++l)
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t color = (layer_count + 3 * l) * n + i;
            double* rgba = output.data() + 4 * (l * n + i);
            rgba[0] = solved[color];
            rgba[1] = solved[color + n];
            rgba[2] = solved[color + 2 * n];
            rgba[3] = solved[l * n + i];
        }
    return true;
}

} // namespace gpu
} // namespace unblending
