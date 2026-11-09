#include <unblending/unblending.hpp>
#include <QImage>

namespace unblending
{
    void AbstractImage::save(const std::string &file_path) const
    {
        QImage q_image(width(), height(), QImage::Format_ARGB32);
        for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
        {
            const IntColor color = get_color(x, y);
            q_image.setPixel(x, y, qRgba(color(0), color(1), color(2), color(3)));
        }
        q_image.save(QString::fromStdString(file_path));
    }

    ColorImage::ColorImage(const std::string &file_path)
    {
        QImage q_image(QString::fromStdString(file_path));
        width_  = q_image.width();
        height_ = q_image.height();

        assert(width() > 0 && height() > 0);

        rgba_ = std::vector<Image>(4, Image(width(), height()));
        for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
        {
            const QRgb q_color = q_image.pixel(x, y);
            rgba_[0].set_pixel(x, y, qRed  (q_color) / 255.0);
            rgba_[1].set_pixel(x, y, qGreen(q_color) / 255.0);
            rgba_[2].set_pixel(x, y, qBlue (q_color) / 255.0);
            rgba_[3].set_pixel(x, y, qAlpha(q_color) / 255.0);
        }
    }

    ColorImage ColorImage::get_scaled_image(int target_width) const
    {
        QImage q_image(width(), height(), QImage::Format_ARGB32);
        for (int x = 0; x < width(); ++ x) for (int y = 0; y < height(); ++ y)
        {
            const IntColor color = get_color(x, y);
            q_image.setPixel(x, y, qRgba(color(0), color(1), color(2), color(3)));
        }

        q_image = q_image.scaledToWidth(target_width, Qt::SmoothTransformation);

        ColorImage new_image(q_image.width(), q_image.height());

        for (int x = 0; x < q_image.width(); ++ x) for (int y = 0; y < q_image.height(); ++ y)
        {
            const QRgb q_rgba = q_image.pixel(x, y);
            const Eigen::Vector4d rgba
            {
                static_cast<double>(qRed  (q_rgba)) / 255.0,
                static_cast<double>(qGreen(q_rgba)) / 255.0,
                static_cast<double>(qBlue (q_rgba)) / 255.0,
                static_cast<double>(qAlpha(q_rgba)) / 255.0
            };
            new_image.set_rgba(x, y, rgba);
        }

        return new_image;
    }

    void export_layers(const std::vector<ColorImage>& layers,
                       const std::string&             output_directory_path,
                       const std::string&             file_name_prefix,
                       const bool                     with_alpha_channel,
                       const bool                     with_blend_mode_suffix,
                       const std::vector<LayerInfo>&  layer_infos)
    {
        assert((!with_blend_mode_suffix) || layer_infos.size() == layers.size());

        for (int index = 0; index < layers.size(); ++ index)
        {
            const std::string suffix = with_blend_mode_suffix ? "_" + retrieve_name(layer_infos[index].blend_mode) : "";
            layers[index].save(output_directory_path + "/" + file_name_prefix + "_" + std::to_string(index) + suffix + ".png");
            if (with_alpha_channel)
            {
                layers[index].get_a().save(output_directory_path + "/" + file_name_prefix + "-alpha_" + std::to_string(index) + ".png");
            }
        }
    }

    void export_models(const std::vector<ColorModelPtr>& models,
                       const std::string&           output_directory_path,
                       const std::string&           file_name_prefix)
    {
        for (int index = 0; index < models.size(); ++ index)
        {
            models[index]->generate_visualization().save(output_directory_path + "/" + file_name_prefix + "_" + std::to_string(index) + ".png");
        }
    }
}
