#include <unblending/unblending.hpp>
#include "gpu_dispatch.hpp"
#include <unblending/image_processing.hpp>
#include <unblending/color_model.hpp>
#include <unblending/equations.hpp>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <nlopt-util.hpp>
#include <timer.hpp>
#include <parallel-util.hpp>

//#define SPARCITY
//#define VERBOSE
//#define MATTE_ALPHA_FORCE_FIXED
//#define AKSOY_INITIAL_SOLUTION
//#define SECOND_LAYER_GRAY
//#define USE_ORIGINAL_HYPERPARAMETERS

namespace unblending
{
    using std::vector;

    namespace
    {
        gpu::Options gpu_options(bool opaque, bool smooth = false)
        {
            gpu::Options options;
            options.opaque = opaque;
            options.smooth = smooth;
#ifdef SPARCITY
            options.sparsity = true;
#endif
#ifdef SECOND_LAYER_GRAY
            options.gray_layer = 1;
#endif
#if defined(AKSOY_INITIAL_SOLUTION) || defined(USE_ORIGINAL_HYPERPARAMETERS)
            options.supported_build = false;
#endif
            return options;
        }
    }

    
    struct OptimizationParameterSet
    {
        vector<ColorModelPtr> models;
        vector<CompOp>        comp_ops;
        vector<BlendMode>     modes;
        
        Vec3   target_color;
        VecX   lambda;
        double rho;
        double sigma;               // Weight for the sparcity term
        bool   use_sparcity;
        bool   use_minimum_alpha;
        bool   use_target_alphas;   // If true, the alternative constraint (Eq. 6) will be used instead of the unity constraint (Eq. 2).
        VecX   target_alphas;       // This will be used when "use_target_alphas" is true.
        
        vector<int> gray_layers;    // A list of gray layer indices. For example, if the second and fourth layers are to be gray, it looks like { 1, 3 }.
    };
    
    double objective_function(const vector<double> &x, vector<double>& grad, void* data)
    {
        const int num_layers = static_cast<int>(x.size() / 4);
        
        const OptimizationParameterSet& set = *static_cast<const OptimizationParameterSet*>(data);
        
        const VecX alphas = Eigen::Map<const VecX>(&x[0], num_layers);
        const VecX colors = Eigen::Map<const VecX>(&x[num_layers], num_layers * 3);
        
        const VecX constraint_vector = calculate_constraint_vector(alphas,
                                                                   colors,
                                                                   set.target_color,
                                                                   set.comp_ops,
                                                                   set.modes,
                                                                   set.use_target_alphas,
                                                                   set.target_alphas,
                                                                   set.gray_layers);
        
        const VecX derivative_of_unmixing_energy   = calculate_derivative_of_unmixing_energy(alphas,
                                                                                             colors,
                                                                                             set.models,
                                                                                             set.sigma,
                                                                                             set.use_sparcity,
                                                                                             set.use_minimum_alpha);
        const MatX derivative_of_constraint_vector = calculate_derivative_of_constraint_vector(alphas,
                                                                                               colors,
                                                                                               set.target_color,
                                                                                               set.comp_ops,
                                                                                               set.modes,
                                                                                               set.use_target_alphas,
                                                                                               set.target_alphas,
                                                                                               set.gray_layers);
        
        const VecX gradient = derivative_of_unmixing_energy + derivative_of_constraint_vector * (set.rho * constraint_vector - set.lambda);
        
        Eigen::Map<VecX>(&grad[0], num_layers * 4) = gradient;
        
        const double unmixing_energy = calculate_unmixing_energy_term(alphas,
                                                                      colors,
                                                                      set.models,
                                                                      set.sigma,
                                                                      set.use_sparcity,
                                                                      set.use_minimum_alpha);
        const double lagrange        = calculate_lagrange_term(constraint_vector, set.lambda);
        const double penalty         = calculate_penalty_term(constraint_vector, set.rho);
        
        return unmixing_energy + lagrange + penalty;
    }
    
    VecX find_initial_solution(const Vec3&                  target_color,
                               const vector<ColorModelPtr>& models)
    {
        const int num_layers = static_cast<int>(models.size());
        
        VecX x_initial = VecX::Zero(num_layers * 4);
        
#ifdef AKSOY_INITIAL_SOLUTION
        double min_distance  = DBL_MAX;
        int    closest_index = - 1;
        for (int index = 0; index < num_layers; ++ index)
        {
            double distance = models[index]->calculate_distance(target_color);
            if (min_distance > distance)
            {
                min_distance  = distance;
                closest_index = index;
            }
        }
        
        // Set the alpha value of the closest layer as 1.0 (and keep those of the other layers as 0.0)
        x_initial(closest_index) = 1.0;
        
        for (int index = 0; index < num_layers; ++ index)
        {
            x_initial.segment<3>(num_layers + index * 3) = (index == closest_index) ? target_color : models[index]->get_representative_color();
            for (int i : { 0, 1, 2}) { x_initial(num_layers + index * 3 + i) = crop_value(x_initial(num_layers + index * 3 + i)); }
        }
#else
        x_initial.segment(0, num_layers) = VecX::Constant(num_layers, 0.50);
        for (int index = 0; index < num_layers; ++ index)
        {
            x_initial.segment<3>(num_layers + index * 3) = models[index]->get_representative_color();
        }
#endif
        
        return x_initial;
    }
    
    VecX solve_per_pixel_optimization(const Vec3&                  target_color,
                                      const vector<ColorModelPtr>& models,
                                      const vector<CompOp>&        comp_ops,
                                      const vector<BlendMode>&     modes,
                                      const bool                   is_for_refinement       = false,
                                      const bool                   has_opaque_background   = true,
                                      const VecX&                  initial_colors          = VecX(),
                                      const VecX&                  target_alphas           = VecX(),
                                      const bool                   force_smooth_background = false,
                                      const Vec3&                  target_background_color = Vec3())
    {
        const int num_layers = static_cast<int>(models.size());
        
#ifdef SECOND_LAYER_GRAY
        const vector<int> gray_layers = { 1 };
#else
        const vector<int> gray_layers;
#endif
        
        VecX upper = VecX::Constant(num_layers * 4, 1.0);
        VecX lower = VecX::Constant(num_layers * 4, 0.0);
        
#ifdef MATTE_ALPHA_FORCE_FIXED
        if (is_for_refinement)
        {
            upper.segment(0, num_layers) = target_alphas;
            lower.segment(0, num_layers) = target_alphas;
        }
#endif
        
        // Find an initial solution
        VecX x = find_initial_solution(target_color, models);
        if (is_for_refinement)
        {
            x.segment(0, num_layers)              = target_alphas;
            x.segment(num_layers, num_layers * 3) = initial_colors;
        }
        
        // Enforce background opacity
        if (has_opaque_background)
        {
            lower(0) = 1.0;
            x(0)     = 1.0;
        }
        
        // Enforce background smoothness
        if (force_smooth_background)
        {
            assert(has_opaque_background);
            
            upper.segment<3>(num_layers) = target_background_color;
            lower.segment<3>(num_layers) = target_background_color;
            x.segment<3>(num_layers)     = target_background_color;
        }
        
        const int num_alpha_constraints = is_for_refinement ? num_layers : 1;
        const int num_constraints       = 3 + num_alpha_constraints + 3 * gray_layers.size();
        
#ifdef USE_ORIGINAL_HYPERPARAMETERS
        constexpr double gamma         = 0.25;
        constexpr double epsilon       = 1e-05;
        constexpr double local_epsilon = 1e-05;
        constexpr double beta          = 10.0;
        constexpr double initial_rho   = 0.1;
#else
        constexpr double gamma         = 0.25;
        constexpr double epsilon       = 5e-03;
        constexpr double local_epsilon = 5e-03;
        constexpr double beta          = 10.0;
        constexpr double initial_rho   = 100.0;
#endif
        
        OptimizationParameterSet set;
        set.models            = models;
        set.comp_ops          = comp_ops;
        set.modes             = modes;
        set.lambda            = VecX::Constant(num_constraints, 0.0);
        set.rho               = initial_rho;
        set.target_color      = target_color;
        set.sigma             = 10.0;
        set.target_alphas     = target_alphas;
#ifdef SPARCITY
        set.use_sparcity      = !is_for_refinement;
#else
        set.use_sparcity      = false;
#endif
        set.use_minimum_alpha = !is_for_refinement;
        set.use_target_alphas = is_for_refinement;
        set.gray_layers       = gray_layers;
        
        int count = 0;
        constexpr int max_count = 20;
        while (true)
        {
            const VecX x_new = nloptutil::solve(x, upper, lower, objective_function, nlopt::LD_LBFGS, &set, false, 1000, local_epsilon, local_epsilon);
            const VecX g     = calculate_constraint_vector(x,
                                                           set.target_color,
                                                           comp_ops,
                                                           modes,
                                                           set.use_target_alphas,
                                                           target_alphas,
                                                           gray_layers);
            const VecX g_new = calculate_constraint_vector(x_new,
                                                           set.target_color,
                                                           comp_ops,
                                                           modes,
                                                           set.use_target_alphas,
                                                           target_alphas,
                                                           gray_layers);
            
            set.lambda -= set.rho * g_new;
            if (g_new.norm() > gamma * g.norm()) { set.rho *= beta; }
            
            const bool is_unchanged = (x_new - x).norm() < epsilon;
            const bool is_satisfied = g_new.norm() < epsilon;
            
            x = x_new;
            
            if ((is_unchanged && is_satisfied) || count > max_count) break;
            
            ++ count;
        }
        
#ifdef VERBOSE
        const VecX g = calculate_constraint_vector(x,
                                                   set.target_color,
                                                   comp_ops,
                                                   modes,
                                                   set.use_target_alphas,
                                                   target_alphas,
                                                   gray_layers);
        if (g.norm() > 0.01)
        {
            const auto comp = composite_layers(x.segment(0, num_layers), x.segment(num_layers, num_layers * 3), comp_ops, modes);
            
            std::cout << "==== Failed to satisfy hard constraints ====" << std::endl;
            std::cout << "count  : " << count << std::endl;
            std::cout << "target : " << target_color.transpose().format(get_format()) << std::endl;
            std::cout << "comp.  : " << comp.transpose().format(get_format()) << std::endl;
            for (int i = 0; i < num_layers; ++ i)
            {
                std::cout << "layer " << i << ": " << x.segment<3>(num_layers + i * 3).transpose().format(get_format()) << ", " << x(i);
                if (is_for_refinement)
                {
                    std::cout << " \t (" << target_alphas(i) << ")";
                }
                std::cout << std::endl;
            }
            std::cout << "g.norm : " << g.norm() << std::endl;
            std::cout << "g      : " << g.transpose().format(get_format()) << std::endl;
        }
#endif
        
        return x;
    }
    
    VecX normalize_alphas(const VecX&           alphas,
                          const vector<CompOp>& comp_ops)
    {
        bool is_all_plus        = true;
        bool is_all_source_over = true;
        for (auto& comp_op : comp_ops)
        {
            is_all_plus        = is_all_plus        && comp_op.is_plus();
            is_all_source_over = is_all_source_over && comp_op.is_source_over();
        }
        
        assert(is_all_plus || is_all_source_over);
        
        constexpr double epsilon = 1e-05;
        
        const bool has_opaque_background = std::abs(alphas(0) - 1.0) < epsilon;
        
        if (is_all_source_over && has_opaque_background) { return alphas; }
        if (is_all_plus) { return alphas / alphas.sum(); }
        
        // This line is never performed
        return VecX::Zero(alphas.rows());
    }
    
    static vector<ColorImage> refine_cpu(const ColorImage& image,
                                         const vector<ColorImage>& layers,
                                         const vector<LayerInfo>& layer_infos,
                                         bool has_opaque_background,
                                         bool force_smooth_background,
                                         int target_concurrency)
    {
        const vector<ColorModelPtr> models                  = extract_color_models           (layer_infos);
        const vector<CompOp>        comp_ops                = extract_comp_ops               (layer_infos);
        const vector<BlendMode>     modes                   = extract_blend_modes            (layer_infos);
        
        assert(layers.size() == models.size());
        
        const int number = static_cast<int>(layers.size());
        const int width  = image.width();
        const int height = image.height();
        const int radius = 60 * std::min(width, height) / 1000;
        
        constexpr double epsilon = 1e-04;
        
        // Apply guided filter
        vector<Image> refined_alphas;
        for (const ColorImage& layer : layers)
        {
            const Image& alpha         = layer.get_a();
            const Image  refined_alpha = apply_guided_filter(alpha, image, radius, epsilon);
            
            refined_alphas.push_back(refined_alpha);
        }
        
        // Crop alphas into [0, 1]
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            for (int i = 0; i < number; ++ i)
            {
                refined_alphas[i].set_pixel(x, y, crop_value(refined_alphas[i].get_pixel(x, y)));
            }
        }
        
        // Normalize alphas such that the composited alpha becomes one for each pixel
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            VecX alphas = VecX(number);
            for (int i = 0; i < number; ++ i)
            {
                alphas(i) = refined_alphas[i].get_pixel(x, y);
            }
            
            alphas = normalize_alphas(alphas, comp_ops);
            
            for (int i = 0; i < number; ++ i)
            {
                refined_alphas[i].set_pixel(x, y, alphas(i));
            }
        }
        
        // Smooth background
        ColorImage smoothed_background(width, height);
        if (force_smooth_background)
        {
            assert(has_opaque_background);
            smoothed_background.set_r(apply_guided_filter(layers[0].get_r(), image, radius, epsilon));
            smoothed_background.set_g(apply_guided_filter(layers[0].get_g(), image, radius, epsilon));
            smoothed_background.set_b(apply_guided_filter(layers[0].get_b(), image, radius, epsilon));
        }
        
        // Perform optimization
        vector<ColorImage> refined_layers(number, ColorImage(width, height));
        auto per_pixel_process = [&](int x, int y)
        {
            VecX initial_colors(number * 3);
            VecX target_alphas(number);
            for (int i = 0; i < number; ++ i)
            {
                initial_colors.segment<3>(i * 3) = layers[i].get_rgb(x, y);
                target_alphas(i)                 = refined_alphas[i].get_pixel(x, y);
            }
            
            if (force_smooth_background)
            {
                initial_colors.segment<3>(0) = crop_vec3(smoothed_background.get_rgb(x, y));
            }
            
            const Vec3 pixel_color = image.get_rgb(x, y);
            const VecX solution = solve_per_pixel_optimization(pixel_color,
                                                               models,
                                                               comp_ops,
                                                               modes,
                                                               true,
                                                               has_opaque_background,
                                                               initial_colors,
                                                               target_alphas,
                                                               force_smooth_background,
                                                               crop_vec3(smoothed_background.get_rgb(x, y)));
            
            for (int index = 0; index < number; ++ index)
            {
                refined_layers[index].set_rgba(x, y, solution.segment<3>(number + index * 3), solution(index));
            }
        };
        
        parallelutil::parallel_for_2d(width, height, per_pixel_process, target_concurrency);
        
        return refined_layers;
    }
    
    static vector<ColorImage> unmix_cpu(const ColorImage& image,
                                        const vector<LayerInfo>& layer_infos,
                                        bool has_opaque_background,
                                        int target_concurrency)
    {
        const vector<ColorModelPtr> models                  = extract_color_models           (layer_infos);
        const vector<CompOp>        comp_ops                = extract_comp_ops               (layer_infos);
        const vector<BlendMode>     modes                   = extract_blend_modes            (layer_infos);
        
        const int width      = image.width();
        const int height     = image.height();
        const int num_layers = static_cast<int>(models.size());
        
        vector<ColorImage> layers(num_layers, ColorImage(width, height));
        
        auto per_pixel_process = [&](int x, int y)
        {
            const Vec3 pixel_color = image.get_rgb(x, y);
            const VecX solution = solve_per_pixel_optimization(pixel_color,
                                                               models,
                                                               comp_ops,
                                                               modes,
                                                               false,
                                                               has_opaque_background);
            
            for (int index = 0; index < num_layers; ++ index)
            {
                layers[index].set_rgba(x, y, solution.segment<3>(num_layers + index * 3), solution(index));
            }
        };
        
        parallelutil::parallel_for_2d(width, height, per_pixel_process, target_concurrency);
        
        return layers;
    }
    
    vector<ColorImage> compute_color_unmixing(const ColorImage& image,
                                             const vector<LayerInfo>& layer_infos,
                                             bool has_opaque_background,
                                             int target_concurrency)
    {
        timer::Timer timer("compute_color_unmixing");
        vector<ColorImage> output;
        if (gpu::try_solve(image, layer_infos, nullptr, gpu_options(has_opaque_background), output))
            return output;
        return unmix_cpu(image, layer_infos, has_opaque_background, target_concurrency);
    }

    vector<ColorImage> perform_matte_refinement(const ColorImage& image,
                                               const vector<ColorImage>& layers,
                                               const vector<LayerInfo>& layer_infos,
                                               bool has_opaque_background,
                                               bool force_smooth_background,
                                               int target_concurrency)
    {
        timer::Timer timer("perform_matte_refinement");
        vector<ColorImage> output;
        if (gpu::try_solve(image, layer_infos, &layers,
                           gpu_options(has_opaque_background, force_smooth_background), output))
            return output;
        return refine_cpu(image, layers, layer_infos, has_opaque_background,
                          force_smooth_background, target_concurrency);
    }

    namespace
    {
        // Normalize the stride once and return the exact size of a packed RGBA stack.
        std::size_t validate_pixels(RgbImageView& image, const vector<LayerInfo>& layers)
        {
            constexpr auto max_int = std::numeric_limits<int>::max();
            if (!image.data || image.width <= 0 || image.height <= 0 ||
                image.width > max_int / image.height)
                throw std::invalid_argument("Invalid RGB image dimensions or data");

            const std::size_t max_doubles = vector<double>().max_size();
            if (std::size_t(image.width) > max_doubles / 3)
                throw std::length_error("RGB row is too large");

            const std::size_t row_size = 3 * std::size_t(image.width);
            if (!image.row_stride) image.row_stride = row_size;
            if (image.row_stride < row_size || image.size < row_size ||
                std::size_t(image.height - 1) > (image.size - row_size) / image.row_stride)
                throw std::invalid_argument("RGB buffer is too small or has an invalid row stride");

            if (layers.empty() || std::any_of(layers.begin(), layers.end(), [](const LayerInfo& layer) {
                    return !layer.color_model;
                }))
                throw std::invalid_argument("Each layer requires a color model");

            const std::size_t pixels = std::size_t(image.width) * image.height;
            if (layers.size() > std::size_t(max_int) / 4 || pixels > max_doubles / 4 / layers.size())
                throw std::length_error("RGBA layer stack is too large");
            return 4 * pixels * layers.size();
        }

        ColorImage read_rgb(RgbImageView image)
        {
            ColorImage result(image.width, image.height);
            for (int y = 0; y < image.height; ++y)
                for (int x = 0; x < image.width; ++x)
                    result.set_rgb(x, y, Eigen::Map<const Vec3>(
                        image.data + std::size_t(y) * image.row_stride + 3 * std::size_t(x)));
            return result;
        }

        vector<ColorImage> read_layers(RgbaLayersView layers, int width, int height, std::size_t count)
        {
            vector<ColorImage> result;
            result.reserve(count);
            const double* pixel = layers.data;
            for (std::size_t l = 0; l < count; ++l)
            {
                result.emplace_back(width, height);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x, pixel += 4)
                        result.back().set_rgba(x, y, Eigen::Map<const Vec4>(pixel));
            }
            return result;
        }

        vector<double> pack_layers(const vector<ColorImage>& layers, std::size_t size)
        {
            vector<double> output(size);
            double* pixel = output.data();
            for (const auto& layer : layers)
                for (int y = 0; y < layer.height(); ++y)
                    for (int x = 0; x < layer.width(); ++x, pixel += 4)
                    {
                        Eigen::Map<Vec4> rgba(pixel);
                        rgba = layer.get_rgba(x, y);
                    }
            return output;
        }
    }

    vector<double> compute_color_unmixing(RgbImageView image,
                                         const vector<LayerInfo>& layer_infos,
                                         bool has_opaque_background,
                                         int target_concurrency)
    {
        const auto size = validate_pixels(image, layer_infos);
        timer::Timer timer("compute_color_unmixing");
        vector<double> output;
        if (gpu::try_solve_pixels(image, layer_infos, nullptr,
                                  gpu_options(has_opaque_background), output))
            return output;
        return pack_layers(unmix_cpu(read_rgb(image), layer_infos,
                                     has_opaque_background, target_concurrency), size);
    }

    vector<double> perform_matte_refinement(RgbImageView image, RgbaLayersView layers,
                                           const vector<LayerInfo>& layer_infos,
                                           bool has_opaque_background,
                                           bool force_smooth_background,
                                           int target_concurrency)
    {
        const auto size = validate_pixels(image, layer_infos);
        if (!layers.data || layers.size != size)
            throw std::invalid_argument("Refinement requires one packed RGBA image per layer");
        if (force_smooth_background && !has_opaque_background)
            throw std::invalid_argument("Smoothing requires an opaque background");
        const auto comp_ops = extract_comp_ops(layer_infos);
        if (!std::all_of(comp_ops.begin(), comp_ops.end(), [](const CompOp& op) { return op.is_plus(); }) &&
            !std::all_of(comp_ops.begin(), comp_ops.end(), [](const CompOp& op) { return op.is_source_over(); }))
            throw std::invalid_argument("Refinement requires an all-Plus or all-SourceOver stack");

        timer::Timer timer("perform_matte_refinement");
        vector<double> output;
        if (gpu::try_solve_pixels(image, layer_infos, &layers,
                                  gpu_options(has_opaque_background, force_smooth_background), output))
            return output;
        return pack_layers(refine_cpu(read_rgb(image),
                                      read_layers(layers, image.width, image.height, layer_infos.size()),
                                      layer_infos, has_opaque_background, force_smooth_background,
                                      target_concurrency), size);
    }

    ColorImage composite_layers(const vector<ColorImage>& layers,
                                const vector<CompOp>&     comp_ops,
                                const vector<BlendMode>&  modes)
    {
        const int number = static_cast<int>(layers.size());
        const int width  = layers.front().width();
        const int height = layers.front().height();
        
        ColorImage composited_image(width, height);
        
        for (int x = 0; x < width; ++ x) for (int y = 0; y < height; ++ y)
        {
            VecX alphas(number);
            VecX colors(number * 3);
            
            for (int index = 0; index < number; ++ index)
            {
                alphas(index)                = layers[index].get_rgba(x, y)(3);
                colors.segment<3>(index * 3) = layers[index].get_rgb(x, y);
            }
            
            const Vec4 composited_color = composite_layers(alphas, colors, comp_ops, modes);
            composited_image.set_rgba(x, y, composited_color);
        }
        
        return composited_image;
    }
}
