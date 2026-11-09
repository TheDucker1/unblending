#pragma once

#include <vector>

namespace unblending {
class ColorImage;
struct LayerInfo;

namespace gpu {
struct Layer {
    double mean[3], precision[9];
    int x, y, z, mode;
};

struct Options {
    bool opaque = true;
    bool smooth = false;
    bool sparsity = false;
    int gray_layer = -1;
    bool supported_build = true;
};

// False leaves output untouched and lets the caller use the original CPU path.
bool try_solve(const ColorImage& image, const std::vector<LayerInfo>& layers,
               const std::vector<ColorImage>* first, Options options,
               std::vector<ColorImage>& output);

#ifdef UNBLENDING_HAS_CUDA
void require_device();
std::vector<double> solve_cuda(int width, int height, const std::vector<Layer>& layers,
                               const std::vector<double>& rgb, const std::vector<double>* first,
                               Options options);
#endif
} // namespace gpu
} // namespace unblending
