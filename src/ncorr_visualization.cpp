#include "ncorr.h"

#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ncorr {

namespace details {

cv::Mat cv_ncorr_data_over_img(const Image2D &img,
                               const Data2D &data,
                               double alpha,
                               double min_data,
                               double max_data,
                               bool enable_colorbar = true,
                               bool enable_axes = true,
                               bool enable_scalebar = true,
                               const std::string &units = "pixels",
                               double units_per_pixel = 1.0,
                               double num_units = -1.0,
                               double font_size = 1.0,
                               ROI2D::difference_type num_tick_marks = 11,
                               int colormap = cv::COLORMAP_JET) {
    typedef ROI2D::difference_type difference_type;

    if (alpha < 0 || alpha > 1) {
        throw std::invalid_argument("alpha input for cv_ncorr_data_over_img() must be between 0 and 1.");
    }
    if (units_per_pixel <= 0) {
        throw std::invalid_argument("units_per_pixel input for cv_ncorr_data_over_img() must be greater than 0.");
    }
    if (num_units != -1 && num_units <= 0) {
        throw std::invalid_argument("num_units input for cv_ncorr_data_over_img() must be greater than 0.");
    }
    if (font_size <= 0) {
        throw std::invalid_argument("font_size input for cv_ncorr_data_over_img() must be greater than 0.");
    }
    if (num_tick_marks < 2) {
        throw std::invalid_argument("num_tick_marks input for cv_ncorr_data_over_img() must be greater than or equal to 2.");
    }

    auto A_img = img.get_gs();
    cv::Mat cv_img = get_cv_img(A_img, min(A_img), max(A_img));
    cv::resize(cv_img, cv_img, { int(data.data_width()), int(data.data_height()) });

    cv::Mat cv_data = get_cv_img(data.get_array(), min_data, max_data);
    cv::applyColorMap(cv_data, cv_data, colormap);

    difference_type border = 20;
    difference_type plot_height = data.data_height() + 2 * border;
    difference_type plot_width = data.data_width() + 2 * border;
    cv::Mat cv_plot(plot_height, plot_width, CV_8UC3, cv::Vec3b(255, 255, 255));

    for (difference_type p2 = border; p2 < data.data_width() + border; ++p2) {
        const difference_type p2_data = p2 - border;
        for (difference_type p1 = border; p1 < data.data_height() + border; ++p1) {
            const difference_type p1_data = p1 - border;

            const auto data_rgb = cv_data.at<cv::Vec3b>(p1_data, p2_data);
            const auto img_gs = cv_img.at<uchar>(p1_data, p2_data);

            if (data.get_roi()(p1_data, p2_data)) {
                cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(img_gs * (1 - alpha) + alpha * data_rgb.val[0],
                                                          img_gs * (1 - alpha) + alpha * data_rgb.val[1],
                                                          img_gs * (1 - alpha) + alpha * data_rgb.val[2]);
            } else {
                cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(img_gs, img_gs, img_gs);
            }
        }
    }

    if (enable_colorbar) {
        auto font_face = cv::FONT_HERSHEY_PLAIN;
        int font_thickness = 1;
        int baseline = 0;

        difference_type num_chars = 8;
        difference_type text_offset_left = 5;
        difference_type colorbar_width = 20;
        difference_type colorbar_bg_width = 0;
        for (difference_type num_mark = 0; num_mark < num_tick_marks; ++num_mark) {
            const auto text_width = cv::getTextSize(std::to_string(double(num_tick_marks - num_mark - 1) / (num_tick_marks - 1) * (min_data - max_data) + max_data).substr(0, num_chars), font_face, 0.75 * font_size, font_thickness, &baseline).width;
            if (text_width > colorbar_bg_width) {
                colorbar_bg_width = text_width;
            }
        }
        colorbar_bg_width += colorbar_width + text_offset_left + border;

        cv::Mat cv_colorbar(plot_height, colorbar_bg_width, cv::DataType<uchar>::type);
        for (difference_type p2 = 0; p2 < colorbar_width; ++p2) {
            for (difference_type p1 = border; p1 < data.data_height() + border; ++p1) {
                cv_colorbar.at<uchar>(p1, p2) = double((data.data_height() + border - 1) - p1) / (data.data_height() - 1) * 255;
            }
        }
        cv::applyColorMap(cv_colorbar, cv_colorbar, colormap);

        cv_colorbar(cv::Range(0, border), cv::Range::all()) = cv::Vec3b(255, 255, 255);
        cv_colorbar(cv::Range(data.data_height() + border, data.data_height() + 2 * border), cv::Range::all()) = cv::Vec3b(255, 255, 255);
        cv_colorbar(cv::Range::all(), cv::Range(colorbar_width, colorbar_bg_width)) = cv::Vec3b(255, 255, 255);

        cv_colorbar(cv::Range(border, border + 1), cv::Range(0, colorbar_width)) = cv::Vec3b(0, 0, 0);
        cv_colorbar(cv::Range(data.data_height() + border - 1, data.data_height() + border), cv::Range(0, colorbar_width)) = cv::Vec3b(0, 0, 0);
        cv_colorbar(cv::Range(border, data.data_height() + border), cv::Range(0, 1)) = cv::Vec3b(0, 0, 0);
        cv_colorbar(cv::Range(border, data.data_height() + border), cv::Range(colorbar_width - 1, colorbar_width)) = cv::Vec3b(0, 0, 0);

        difference_type tick_mark_width = 4;
        for (difference_type num_mark = 0; num_mark < num_tick_marks; ++num_mark) {
            const difference_type p1 = (num_mark * (data.data_height() - 1)) / (num_tick_marks - 1) + border;

            cv_colorbar(cv::Range(p1, p1 + 1), cv::Range(0, tick_mark_width)) = cv::Vec3b(0, 0, 0);
            cv_colorbar(cv::Range(p1, p1 + 1), cv::Range(colorbar_width - tick_mark_width, colorbar_width)) = cv::Vec3b(0, 0, 0);

            std::string tick_mark_label_str = std::to_string(num_mark * (min_data - max_data) / (num_tick_marks - 1) + max_data).substr(0, num_chars);
            auto text_size = cv::getTextSize(tick_mark_label_str, font_face, 0.75 * font_size, font_thickness, &baseline);
            cv::putText(cv_colorbar,
                        tick_mark_label_str,
                        cv::Point(colorbar_width + text_offset_left, num_mark * (data.data_height() - 1) / (num_tick_marks - 1) + border + text_size.height / 2),
                        font_face,
                        0.75 * font_size,
                        cv::Scalar::all(0),
                        font_thickness);
        }

        cv::hconcat(cv_plot, cv_colorbar, cv_plot);
    }

    if (enable_axes) {
        const difference_type axes_length = 0.25 * std::min(data.data_height(), data.data_width());

        cv::Point bg_axes_pts[1][8];
        bg_axes_pts[0][0] = cv::Point(border, border);
        bg_axes_pts[0][1] = cv::Point(border, border + axes_length);
        bg_axes_pts[0][2] = cv::Point(border + 0.20 * axes_length, border + 0.80 * axes_length);
        bg_axes_pts[0][3] = cv::Point(border + 0.10 * axes_length, border + 0.80 * axes_length);
        bg_axes_pts[0][4] = cv::Point(border + 0.10 * axes_length, border + 0.10 * axes_length);
        bg_axes_pts[0][5] = cv::Point(border + 0.80 * axes_length, border + 0.10 * axes_length);
        bg_axes_pts[0][6] = cv::Point(border + 0.80 * axes_length, border + 0.20 * axes_length);
        bg_axes_pts[0][7] = cv::Point(border + axes_length, border);
        const cv::Point* pts1[1] = { bg_axes_pts[0] };
        int npts1[] = { 8 };
        cv::fillPoly(cv_plot, pts1, npts1, 1, cv::Scalar(255, 255, 255));

        cv::Point fg_axes_pts[1][8];
        fg_axes_pts[0][0] = cv::Point(border, border);
        fg_axes_pts[0][1] = cv::Point(border, border + 0.95 * axes_length);
        fg_axes_pts[0][2] = cv::Point(border + 0.125 * axes_length, border + 0.825 * axes_length);
        fg_axes_pts[0][3] = cv::Point(border + 0.070 * axes_length, border + 0.825 * axes_length);
        fg_axes_pts[0][4] = cv::Point(border + 0.070 * axes_length, border + 0.070 * axes_length);
        fg_axes_pts[0][5] = cv::Point(border + 0.825 * axes_length, border + 0.070 * axes_length);
        fg_axes_pts[0][6] = cv::Point(border + 0.825 * axes_length, border + 0.125 * axes_length);
        fg_axes_pts[0][7] = cv::Point(border + 0.95 * axes_length, border);
        const cv::Point* pts2[1] = { fg_axes_pts[0] };
        int npts2[] = { 8 };
        cv::fillPoly(cv_plot, pts2, npts2, 1, cv::Scalar(0, 0, 0));

        const difference_type label_border = 5;
        const difference_type label_offset = 5;
        const double label_alpha = 0.5;

        auto font_face = cv::FONT_HERSHEY_PLAIN;
        int font_thickness = 1;
        int baseline = 0;

        std::string y_str = "Y";
        auto text_size_y = cv::getTextSize(y_str, font_face, font_size, font_thickness, &baseline);
        for (difference_type p2 = border + label_offset; p2 < border + label_offset + 2.0 * label_border + text_size_y.width; ++p2) {
            for (difference_type p1 = border + axes_length + label_offset; p1 < border + axes_length + label_offset + 2.0 * label_border + text_size_y.height; ++p1) {
                if (p1 >= 0 && p1 < cv_plot.rows && p2 >= 0 && p2 < cv_plot.cols) {
                    auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1, p2);
                    cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(label_alpha * cv_plot_rgb.val[0],
                                                              label_alpha * cv_plot_rgb.val[1],
                                                              label_alpha * cv_plot_rgb.val[2]);
                }
            }
        }
        cv::putText(cv_plot,
                    y_str,
                    cv::Point(border + label_offset + label_border, border + axes_length + label_offset + label_border + text_size_y.height),
                    font_face,
                    font_size,
                    cv::Scalar::all(255),
                    font_thickness);

        std::string x_str = "X";
        auto text_size_x = cv::getTextSize(x_str, font_face, font_size, font_thickness, &baseline);
        for (difference_type p2 = border + axes_length + label_offset; p2 < border + axes_length + label_offset + 2.0 * label_border + text_size_x.width; ++p2) {
            for (difference_type p1 = border + label_offset; p1 < border + label_offset + 2.0 * label_border + text_size_x.height; ++p1) {
                if (p1 >= 0 && p1 < cv_plot.rows && p2 >= 0 && p2 < cv_plot.cols) {
                    auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1, p2);
                    cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(label_alpha * cv_plot_rgb.val[0],
                                                              label_alpha * cv_plot_rgb.val[1],
                                                              label_alpha * cv_plot_rgb.val[2]);
                }
            }
        }
        cv::putText(cv_plot,
                    x_str,
                    cv::Point(border + axes_length + label_offset + label_border, border + label_offset + label_border + text_size_x.height),
                    font_face,
                    font_size,
                    cv::Scalar::all(255),
                    font_thickness);
    }

    if (enable_scalebar) {
        const difference_type scalebar_width = num_units == -1 ? data.data_width() / 2 : num_units / units_per_pixel / data.get_scalefactor();
        const difference_type scalebar_height = 5;

        if (num_units == -1) {
            num_units = scalebar_width * units_per_pixel * data.get_scalefactor();
        }

        const difference_type decimal_places = 2;
        std::stringstream ss_scalebar;
        ss_scalebar << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << num_units;
        std::string scalebar_str_unformatted = ss_scalebar.str();
        auto idx_period = scalebar_str_unformatted.find(".");
        std::string scalebar_str = idx_period == std::string::npos ? scalebar_str_unformatted : scalebar_str_unformatted.substr(0, idx_period + decimal_places + 1) + " " + units;

        auto font_face = cv::FONT_HERSHEY_PLAIN;
        int font_thickness = 1;
        int baseline = 0;
        auto text_size = cv::getTextSize(scalebar_str, font_face, font_size, font_thickness, &baseline);

        const difference_type scalebar_bg_border = 10;
        const difference_type scalebar_bg_width = scalebar_width + 2 * scalebar_bg_border;
        const difference_type scalebar_bg_height = scalebar_height + 3 * scalebar_bg_border + text_size.height;
        const difference_type scalebar_bg_offset = 10;
        const double scalebar_bg_alpha = 0.5;
        for (difference_type p2 = border + scalebar_bg_offset; p2 < border + scalebar_bg_offset + scalebar_bg_width; ++p2) {
            for (difference_type p1 = border + data.data_height() - scalebar_bg_offset - scalebar_bg_height; p1 < border + data.data_height() - scalebar_bg_offset; ++p1) {
                if (p1 >= 0 && p1 < cv_plot.rows && p2 >= 0 && p2 < cv_plot.cols) {
                    auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1, p2);
                    cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(scalebar_bg_alpha * cv_plot_rgb.val[0],
                                                              scalebar_bg_alpha * cv_plot_rgb.val[1],
                                                              scalebar_bg_alpha * cv_plot_rgb.val[2]);
                }
            }
        }

        for (difference_type p2 = border + scalebar_bg_offset + scalebar_bg_border; p2 < border + scalebar_bg_offset + scalebar_bg_border + scalebar_width; ++p2) {
            for (difference_type p1 = border + data.data_height() - scalebar_bg_offset - scalebar_bg_border - scalebar_height; p1 < border + data.data_height() - scalebar_bg_offset - scalebar_bg_border; ++p1) {
                if (p1 >= 0 && p1 < cv_plot.rows && p2 >= 0 && p2 < cv_plot.cols) {
                    cv_plot.at<cv::Vec3b>(p1, p2) = cv::Vec3b(255, 255, 255);
                }
            }
        }

        cv::putText(cv_plot,
                    scalebar_str,
                    cv::Point(border + scalebar_bg_offset + scalebar_bg_width / 2.0 - text_size.width / 2.0, border + data.data_height() - scalebar_bg_offset - 2 * scalebar_bg_border - scalebar_height),
                    font_face,
                    font_size,
                    cv::Scalar::all(255),
                    font_thickness);
    }

    return cv_plot;
}

} // namespace details

void imshow_ncorr_data_over_img(const Image2D &img, const Data2D &data, ROI2D::difference_type delay) {
    double min_data = 0;
    double max_data = 0;
    Array2D<double> data_values = data.get_array()(data.get_roi().get_mask());
    if (!data_values.empty()) {
        min_data = prctile(data_values, 0.01);
        max_data = prctile(data_values, 0.99);
    }

    auto cv_img = details::cv_ncorr_data_over_img(img, data, 0.5, min_data, max_data, true, true, false);
    cv::imshow("Ncorr data", cv_img);
    delay == -1 ? cv::waitKey() : cv::waitKey(delay);
}

void save_ncorr_data_over_img(const std::string &filename,
                              const Image2D &img,
                              const Data2D &data,
                              double alpha,
                              double min_data,
                              double max_data,
                              bool enable_colorbar,
                              bool enable_axes,
                              bool enable_scalebar,
                              const std::string &units,
                              double units_per_pixel,
                              double num_units,
                              double font_size,
                              ROI2D::difference_type num_tick_marks,
                              int colormap) {
    auto cv_data_img = details::cv_ncorr_data_over_img(img, data, alpha, min_data, max_data, enable_colorbar, enable_axes, enable_scalebar, units, units_per_pixel, num_units, font_size, num_tick_marks, colormap);
    cv::imwrite(filename, cv_data_img);
}

void save_ncorr_data_over_img_video(const std::string &filename,
                                    const std::vector<Image2D> &imgs,
                                    const std::vector<Data2D> &data,
                                    double alpha,
                                    double fps,
                                    double min_data,
                                    double max_data,
                                    bool enable_colorbar,
                                    bool enable_axes,
                                    bool enable_scalebar,
                                    const std::string &units,
                                    double units_per_pixel,
                                    double num_units,
                                    double font_size,
                                    ROI2D::difference_type num_tick_marks,
                                    int colormap,
                                    double end_delay,
                                    int fourcc) {
    typedef ROI2D::difference_type difference_type;

    if (imgs.size() != 1 && imgs.size() != data.size()) {
        throw std::invalid_argument("Number of images used in save_data_over_img_video() must either be 1, or equal to the number of input data.");
    }
    if (data.empty()) {
        throw std::invalid_argument("Number of Data2D used in save_data_over_img_video() must be greater than or equal to 1.");
    }
    for (difference_type data_idx = 1; data_idx < static_cast<difference_type>(data.size()); ++data_idx) {
        if (data[data_idx].data_height() != data.front().data_height() ||
            data[data_idx].data_width() != data.front().data_width()) {
            throw std::invalid_argument("Attempted to use save_data_over_img_video() with data of differing sizes. All data must be the same size.");
        }
    }
    if (fps <= 0) {
        throw std::invalid_argument("fps input for save_data_over_img_video() must be greater than 0.");
    }
    if (end_delay < 0) {
        throw std::invalid_argument("end_delay input for save_data_over_img_video() must be greater than or equal to 0.");
    }

    std::cout << "\nSaving video: " << filename << "..." << std::endl;

    cv::VideoWriter output_video;
    for (difference_type data_idx = 0; data_idx < static_cast<difference_type>(data.size()); ++data_idx) {
        std::cout << "Frame " << data_idx + 1 << " of " << data.size() << "." << std::endl;

        auto cv_data_img = details::cv_ncorr_data_over_img(imgs.size() == 1 ? imgs.front() : imgs[data_idx],
                                                           data[data_idx],
                                                           alpha,
                                                           min_data,
                                                           max_data,
                                                           enable_colorbar,
                                                           enable_axes,
                                                           enable_scalebar,
                                                           units,
                                                           units_per_pixel,
                                                           num_units,
                                                           font_size,
                                                           num_tick_marks,
                                                           colormap);

        if (data_idx == 0) {
            output_video.open(filename, fourcc, fps, { cv_data_img.cols, cv_data_img.rows }, true);
            if (!output_video.isOpened()) {
                throw std::invalid_argument("Cannot open video file: " + filename + " for save_data_over_img_video().");
            }
        }

        output_video << cv_data_img;
    }

    for (difference_type idx = 0; idx < fps * end_delay; ++idx) {
        output_video << details::cv_ncorr_data_over_img(imgs.size() == 1 ? imgs.front() : imgs.back(),
                                                        data.back(),
                                                        alpha,
                                                        min_data,
                                                        max_data,
                                                        enable_colorbar,
                                                        enable_axes,
                                                        enable_scalebar,
                                                        units,
                                                        units_per_pixel,
                                                        num_units,
                                                        font_size,
                                                        num_tick_marks,
                                                        colormap);
    }
}

void save_DIC_video(const std::string &filename,
                    const DIC_analysis_input &DIC_input,
                    const DIC_analysis_output &DIC_output,
                    DISP disp_type,
                    double alpha,
                    double fps,
                    double min_disp,
                    double max_disp,
                    bool enable_colorbar,
                    bool enable_axes,
                    bool enable_scalebar,
                    double num_units,
                    double font_size,
                    ROI2D::difference_type num_tick_marks,
                    int colormap,
                    double end_delay,
                    int fourcc) {
    typedef ROI2D::difference_type difference_type;

    std::function<const Data2D&(const Disp2D&)> get_disp;
    switch (disp_type) {
        case DISP::V:
            get_disp = &Disp2D::get_v;
            break;
        case DISP::U:
            get_disp = &Disp2D::get_u;
            break;
    }

    Array2D<double> data_values_first = get_disp(DIC_output.disps.front()).get_array()(DIC_output.disps.front().get_roi().get_mask());
    Array2D<double> data_values_last = get_disp(DIC_output.disps.back()).get_array()(DIC_output.disps.back().get_roi().get_mask());
    if (std::isnan(min_disp) && !data_values_first.empty() && !data_values_last.empty()) {
        min_disp = std::min(prctile(data_values_first, 0.01), prctile(data_values_last, 0.01));
    }
    if (std::isnan(max_disp) && !data_values_first.empty() && !data_values_last.empty()) {
        max_disp = std::max(prctile(data_values_first, 0.99), prctile(data_values_last, 0.99));
    }

    std::vector<Data2D> data;
    for (difference_type disp_idx = 0; disp_idx < static_cast<difference_type>(DIC_output.disps.size()); ++disp_idx) {
        data.push_back(get_disp(DIC_output.disps[disp_idx]));
    }

    std::vector<Image2D> imgs;
    switch (DIC_output.perspective_type) {
        case PERSPECTIVE::LAGRANGIAN:
            imgs.push_back(DIC_input.imgs.front());
            break;
        case PERSPECTIVE::EULERIAN:
            for (difference_type img_idx = 1; img_idx < static_cast<difference_type>(DIC_input.imgs.size()); ++img_idx) {
                imgs.push_back(DIC_input.imgs[img_idx]);
            }
            break;
    }

    save_ncorr_data_over_img_video(filename, imgs, data, alpha, fps, min_disp, max_disp, enable_colorbar, enable_axes, enable_scalebar, DIC_output.units, DIC_output.units_per_pixel, num_units, font_size, num_tick_marks, colormap, end_delay, fourcc);
}

void save_strain_video(const std::string &filename,
                       const strain_analysis_input &strain_input,
                       const strain_analysis_output &strain_output,
                       STRAIN strain_type,
                       double alpha,
                       double fps,
                       double min_strain,
                       double max_strain,
                       bool enable_colorbar,
                       bool enable_axes,
                       bool enable_scalebar,
                       double num_units,
                       double font_size,
                       ROI2D::difference_type num_tick_marks,
                       int colormap,
                       double end_delay,
                       int fourcc) {
    typedef ROI2D::difference_type difference_type;

    std::function<const Data2D&(const Strain2D&)> get_strain;
    switch (strain_type) {
        case STRAIN::EYY:
            get_strain = &Strain2D::get_eyy;
            break;
        case STRAIN::EXY:
            get_strain = &Strain2D::get_exy;
            break;
        case STRAIN::EXX:
            get_strain = &Strain2D::get_exx;
            break;
    }

    Array2D<double> data_values_first = get_strain(strain_output.strains.front()).get_array()(strain_output.strains.front().get_roi().get_mask());
    Array2D<double> data_values_last = get_strain(strain_output.strains.back()).get_array()(strain_output.strains.back().get_roi().get_mask());
    if (std::isnan(min_strain) && !data_values_first.empty() && !data_values_last.empty()) {
        min_strain = std::min(prctile(data_values_first, 0.01), prctile(data_values_last, 0.01));
    }
    if (std::isnan(max_strain) && !data_values_first.empty() && !data_values_last.empty()) {
        max_strain = std::max(prctile(data_values_first, 0.99), prctile(data_values_last, 0.99));
    }

    std::vector<Data2D> data;
    for (difference_type strain_idx = 0; strain_idx < static_cast<difference_type>(strain_output.strains.size()); ++strain_idx) {
        data.push_back(get_strain(strain_output.strains[strain_idx]));
    }

    std::vector<Image2D> imgs;
    switch (strain_input.DIC_output.perspective_type) {
        case PERSPECTIVE::LAGRANGIAN:
            imgs.push_back(strain_input.DIC_input.imgs.front());
            break;
        case PERSPECTIVE::EULERIAN:
            for (difference_type img_idx = 1; img_idx < static_cast<difference_type>(strain_input.DIC_input.imgs.size()); ++img_idx) {
                imgs.push_back(strain_input.DIC_input.imgs[img_idx]);
            }
            break;
    }

    save_ncorr_data_over_img_video(filename, imgs, data, alpha, fps, min_strain, max_strain, enable_colorbar, enable_axes, enable_scalebar, strain_input.DIC_output.units, strain_input.DIC_output.units_per_pixel, num_units, font_size, num_tick_marks, colormap, end_delay, fourcc);
}

} // namespace ncorr
