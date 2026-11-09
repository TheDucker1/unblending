#ifndef UNBLENDING_PIXELS_HPP
#define UNBLENDING_PIXELS_HPP

#include <cstddef>
#include <vector>
#include <unblending/color_model.hpp>
#include <unblending/layer_info.hpp>

namespace unblending
{
    /// Row-major RGB triples in [0,1]. Size and stride are measured in doubles.
    /// A zero row stride uses tightly packed rows (3 * width).
    struct RgbImageView
    {
        const double* data;
        std::size_t size;
        int width, height;
        std::size_t row_stride;

        RgbImageView(const double* data, std::size_t size, int width, int height,
                     std::size_t row_stride = 0)
            : data(data), size(size), width(width), height(height), row_stride(row_stride) {}
    };

    /// Packed RGBA layer values in [0,1], with size measured in doubles.
    /// Layout is [layer][row][column][channel]. Layers are ordered bottom to top.
    struct RgbaLayersView
    {
        const double* data;
        std::size_t size;
    };

    /// Split RGB pixels into a packed RGBA layer buffer of size 4 * width * height * layer count.
    std::vector<double> compute_color_unmixing(RgbImageView image,
                                               const std::vector<LayerInfo>& layer_infos,
                                               bool has_opaque_background,
                                               int target_concurrency = 0);

    /// Refine a packed RGBA buffer from the first pass, using the original RGB pixels.
    std::vector<double> perform_matte_refinement(RgbImageView image, RgbaLayersView layers,
                                                 const std::vector<LayerInfo>& layer_infos,
                                                 bool has_opaque_background,
                                                 bool force_smooth_background,
                                                 int target_concurrency = 0);
}

#endif // UNBLENDING_PIXELS_HPP
