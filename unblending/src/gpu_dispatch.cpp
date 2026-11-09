#include "gpu_dispatch.hpp"
#include <unblending/unblending.hpp>

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

bool try_solve(const ColorImage& image, const std::vector<LayerInfo>& layers,
               const std::vector<ColorImage>* first, Options options,
               std::vector<ColorImage>& output)
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
        const int w = image.width(), h = image.height();
        // The streaming filter indexes up to sixteen planes with signed ints.
        constexpr size_t max_pixels = (std::numeric_limits<int>::max() - 255) / 16;
        if (w <= 0 || h <= 0 || size_t(w) > max_pixels / size_t(h) ||
            layers.empty() || layers.size() > 4096 ||
            (first && first->size() != layers.size()))
            throw std::invalid_argument("Image or layer dimensions are outside the CUDA solver domain");
        if (!options.supported_build)
            throw std::invalid_argument("This solver build uses CPU-specific initialization or tolerances");

        auto model = describe(layers);
        if (first)
        {
            for (const auto& layer : *first)
                if (layer.width() != w || layer.height() != h)
                    throw std::invalid_argument("Refinement requires layers with the input image dimensions");
        }
        require_device();
        const size_t n = size_t(w) * h, l = layers.size();
        std::vector<double> rgb(3 * n);
        std::vector<double> initial(first ? 4 * l * n : 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                const size_t i = size_t(y) * w + x;
                const auto pixel = image.get_rgb(x, y);
                for (int r = 0; r < 3; ++r) rgb[size_t(r) * n + i] = pixel(r);
            }
        if (first)
            for (size_t layer = 0; layer < l; ++layer)
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                    {
                        const size_t i = size_t(y) * w + x;
                        const auto pixel = (*first)[layer].get_rgba(x, y);
                        initial[layer * n + i] = pixel(3);
                        for (int r = 0; r < 3; ++r)
                            initial[(l + 3 * layer + r) * n + i] = pixel(r);
                    }

        const auto solved = solve_cuda(w, h, model, rgb, first ? &initial : nullptr, options);
        std::vector<ColorImage> result;
        result.reserve(l);
        for (size_t layer = 0; layer < l; ++layer)
        {
            result.emplace_back(w, h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const size_t i = size_t(y) * w + x;
                    const size_t color = (l + 3 * layer) * n + i;
                    result.back().set_rgba(x, y, Vec3(solved[color], solved[color + n], solved[color + 2 * n]),
                                           solved[layer * n + i]);
                }
        }
        output = std::move(result);
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

} // namespace gpu
} // namespace unblending
