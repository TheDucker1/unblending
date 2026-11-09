#pragma once

#include "gpu_dispatch.hpp"
#include <unblending/pixels.hpp>

namespace unblending {
namespace detail {
std::vector<double> unmix_pixels(RgbImageView image, const std::vector<LayerInfo>& layers,
                                 bool opaque, int concurrency, gpu::Backend backend);
std::vector<double> refine_pixels(RgbImageView image, RgbaLayersView first,
                                  const std::vector<LayerInfo>& layers, bool opaque,
                                  bool smooth, int concurrency, gpu::Backend backend);
}
}
