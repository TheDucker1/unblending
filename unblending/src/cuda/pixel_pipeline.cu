#include "../gpu_dispatch.hpp"
#include "solver/solver_workspace.cuh"

namespace unblending {
namespace gpu {
namespace {
template<class T>
class DeviceBuffer {
    T* data_ = nullptr;
public:
    explicit DeviceBuffer(size_t size)
    {
        if (size) ck(cudaMalloc(&data_, size * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() const { return data_; }
    void upload(const std::vector<T>& values)
    {
        ck(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};
} // namespace

void require_device()
{
    int devices = 0;
    ck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("No CUDA device is available");
}

std::vector<double> solve_cuda(int width, int height, const std::vector<Layer>& layers,
                               const std::vector<double>& rgb, const std::vector<double>* first,
                               Options options)
{
    const int n = width * height, l = static_cast<int>(layers.size());
    general::Problem problem{l};
    problem.opaque = options.opaque;
    problem.refinement = first != nullptr;
    problem.smooth = problem.refinement && options.smooth;
    problem.fixed_alpha = problem.refinement;
    problem.minimum = !problem.refinement;
    problem.sparsity = problem.minimum && options.sparsity;

    std::vector<custom_backend::Model> models(l);
    for (int i = 0; i < l; ++i)
    {
        auto& model = models[i];
        const auto& layer = layers[i];
        std::copy_n(layer.mean, 3, model.mean);
        std::copy_n(layer.precision, 9, model.inverse);
        model.X = layer.x;
        model.Y = layer.y;
        model.Z = layer.z;
        model.mode = layer.mode;
        model.gray = i == options.gray_layer;
        custom_backend::metric(model);
    }
    if (problem.refinement)
        general::normalization_kind(models.data(), l, problem.opaque, problem.smooth);

    DeviceBuffer<custom_backend::Model> device_models(l);
    device_models.upload(models);

    const size_t layer_values = size_t(4) * l * n;
    DeviceBuffer<double> target(rgb.size()), initial(layer_values), output(layer_values);
    DeviceBuffer<double> filter_targets(size_t(l + 3) * n), input_layers(first ? layer_values : 0);
    target.upload(rgb);
    const auto alpha = filter_targets.data();
    const auto bg = alpha + size_t(l) * n;

    general::Options solver_options;
    solver_options.starts = 3;
    solver_options.correction = true;
    solver_options.project_gray = true;
    custom_backend::Workspace workspace(n, l, problem.constraints(models.data()), true, 4,
                                        solver_options, size_t(128) * 1024 * 1024, 0, models.data());
    std::unique_ptr<general::StreamingFilterGPU<double>> filter;
    if (first)
    {
        input_layers.upload(*first);
        filter = std::make_unique<general::StreamingFilterGPU<double>>(
            width, height, l, options.opaque, problem.smooth, models.data());
        const int radius = 60 * std::min(width, height) / 1000;
        filter->run(target.data(), input_layers.data(), alpha, radius);
        general::initialize_refinement<<<(n + 127) / 128, 128>>>(
            problem, n, input_layers.data(), alpha, initial.data());
    }
    else
    {
        ck(cudaMemset(alpha, 0, size_t(l + 3) * n * sizeof(double)));
        general::initialize_first<<<(n + 127) / 128, 128>>>(device_models.data(), problem, n, initial.data());
    }
    ck(cudaGetLastError());
    workspace.solve(device_models.data(), problem, initial.data(), target.data(), alpha, bg,
                    output.data(), 8, filter ? filter->device_status() : nullptr);

    std::vector<int> status(n);
    ck(cudaMemcpy(status.data(), workspace.status(), size_t(n) * sizeof(int), cudaMemcpyDeviceToHost));
    if (std::any_of(status.begin(), status.end(), [](int code) { return code != 0; }))
        throw std::runtime_error("GPU solver could not satisfy the image or refinement constraints");

    std::vector<double> solved(layer_values);
    ck(cudaMemcpy(solved.data(), output.data(), layer_values * sizeof(double), cudaMemcpyDeviceToHost));
    return solved;
}

} // namespace gpu
} // namespace unblending
