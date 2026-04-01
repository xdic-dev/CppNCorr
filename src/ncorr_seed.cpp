#include "ncorr.h"
#include "ncorr/algo/seed_analysis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace ncorr {

namespace details {

bool analyze_point(
    const Array2D<double> &queue_params,
    ROI2D::difference_type p1_delta,
    ROI2D::difference_type p2_delta,
    const ROI2D::region_nlinfo &nlinfo,
    ROI2D::difference_type scalefactor,
    const subregion_nloptimizer &sr_nloptimizer,
    double cutoff_corrcoef,
    double cutoff_delta_disp,
    std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&, const Array2D<double>&)>> &queue,
    Array2D<bool> &A_ap,
    Array2D<double> &params_buf
);

} // namespace details

Disp2D RGDIC_without_thread(
    const Array2D<double> &A_ref,
    const Array2D<double> &A_cur,
    const ROI2D &roi,
    ROI2D::difference_type scalefactor,
    INTERP interp_type,
    SUBREGION subregion_type,
    ROI2D::difference_type r,
    double cutoff_corrcoef,
    bool debug,
    const std::vector<SeedParams>& seeds_by_region,
    bool seeds_are_optimized
);

namespace algo {

Disp2D compute_displacements(
    details::subregion_nloptimizer &sr_nloptimizer,
    const ROI2D& roi_reduced,
    const SeedParams& seedparams,
    ROI2D::difference_type scalefactor,
    double cutoff_corrcoef,
    ROI2D::difference_type region_idx,
    bool debug
) {
    const auto H = roi_reduced.height();
    const auto W = roi_reduced.width();
    const double cutoff_delta_disp = scalefactor;

    Array2D<double> A_v(H, W);
    Array2D<double> A_u(H, W);
    Array2D<double> A_cc(H, W);
    Array2D<bool> A_ap(roi_reduced.get_mask());
    Array2D<bool> A_vp(H, W);
    Array2D<double> params_buf(10, 1);

    Array2D<double> seed_params(10, 1);
    seed_params(0) = seedparams.y;
    seed_params(1) = seedparams.x;
    seed_params(2) = seedparams.v;
    seed_params(3) = seedparams.u;
    seed_params(4) = seedparams.dv_dy;
    seed_params(5) = seedparams.dv_dx;
    seed_params(6) = seedparams.du_dy;
    seed_params(7) = seedparams.du_dx;
    seed_params(8) = seedparams.corrcoef;
    seed_params(9) = 0;

    if (!seed_params.empty()) {
        A_ap(seed_params(0) / scalefactor, seed_params(1) / scalefactor) = false;

        auto comp = [](const Array2D<double> &a, const Array2D<double> &b ) { return a(8) > b(8); };
        std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&, const Array2D<double>&)>> queue(comp);

        queue.push(seed_params);
        while (!queue.empty()) {
            auto queue_params = std::move(queue.top());
            queue.pop();

            A_vp(queue_params(0) / scalefactor, queue_params(1) / scalefactor) = true;
            A_v(queue_params(0) / scalefactor, queue_params(1) / scalefactor) = queue_params(2);
            A_u(queue_params(0) / scalefactor, queue_params(1) / scalefactor) = queue_params(3);
            A_cc(queue_params(0) / scalefactor, queue_params(1) / scalefactor) = queue_params(8);

            details::analyze_point(queue_params, -scalefactor, 0, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
            details::analyze_point(queue_params, scalefactor, 0, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
            details::analyze_point(queue_params, 0, -scalefactor, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
            details::analyze_point(queue_params, 0, scalefactor, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
        }
    }

    auto A_vp_buf = A_vp;
    for (ROI2D::difference_type region = 0; region < roi_reduced.size_regions(); ++region) {
        for (ROI2D::difference_type nl_idx = 0; nl_idx < roi_reduced.get_nlinfo(region).nodelist.width(); ++nl_idx) {
            const ROI2D::difference_type p2 = nl_idx + roi_reduced.get_nlinfo(region).left_nl;
            for (ROI2D::difference_type np_idx = 0; np_idx < roi_reduced.get_nlinfo(region).noderange(nl_idx); np_idx += 2) {
                const ROI2D::difference_type np_top = roi_reduced.get_nlinfo(region).nodelist(np_idx, nl_idx);
                const ROI2D::difference_type np_bottom = roi_reduced.get_nlinfo(region).nodelist(np_idx + 1, nl_idx);
                for (ROI2D::difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    if ((roi_reduced.get_nlinfo(region).in_nlinfo(p1 - 1, p2) && A_vp_buf(p1 - 1, p2) && std::sqrt(std::pow(A_v(p1 - 1, p2) - A_v(p1, p2), 2) + std::pow(A_u(p1 - 1, p2) - A_u(p1, p2), 2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region).in_nlinfo(p1 + 1, p2) && A_vp_buf(p1 + 1, p2) && std::sqrt(std::pow(A_v(p1 + 1, p2) - A_v(p1, p2), 2) + std::pow(A_u(p1 + 1, p2) - A_u(p1, p2), 2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region).in_nlinfo(p1, p2 - 1) && A_vp_buf(p1, p2 - 1) && std::sqrt(std::pow(A_v(p1, p2 - 1) - A_v(p1, p2), 2) + std::pow(A_u(p1, p2 - 1) - A_u(p1, p2), 2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region).in_nlinfo(p1, p2 + 1) && A_vp_buf(p1, p2 + 1) && std::sqrt(std::pow(A_v(p1, p2 + 1) - A_v(p1, p2), 2) + std::pow(A_u(p1, p2 + 1) - A_u(p1, p2), 2)) > cutoff_delta_disp)) {
                        A_vp(p1, p2) = false;
                    }
                }
            }
        }
    }

    auto roi_valid = roi_reduced.form_union(A_vp);
    if (debug) {
        std::cout << "RGDIC completed with manual seed" << std::endl;
    }
    return Disp2D(std::move(A_v), std::move(A_u), std::move(A_cc), roi_valid, scalefactor);
}

std::vector<SeedScheduleFrame> compute_only_seed_points(
    const Array2D<double>& A_ref,
    const std::vector<Array2D<double>>& A_curs,
    const ROI2D& roi,
    ROI2D::difference_type scalefactor,
    INTERP interp_type,
    SUBREGION subregion_type,
    ROI2D::difference_type r,
    const std::vector<SeedParams>& seeds_by_region,
    double cutoff_corrcoef,
    ROI2D::difference_type region_idx,
    bool debug
) {
    std::vector<SeedScheduleFrame> selected_frames;

    if (debug) {
        std::cout << "\n=== Compute seed parameters for all frames with ROI updates ===>\n";
        std::cout << "Starting with " << seeds_by_region.size() << " seeds" << std::endl;
    }

    Array2D<double> A_ref_current = A_ref;
    ROI2D roi_current = roi;
    std::vector<SeedParams> seeds_current = seeds_by_region;
    ROI2D::difference_type ref_frame_idx_current = 0;

    std::size_t idx = 0;
    while (idx < A_curs.size()) {
        const auto& A_cur = A_curs[idx];

        if (debug) {
            std::cout << "\n=== Seed analysis frame " << (idx + 1) << " ===>" << std::endl;
        }

        auto sr_nloptimizer = details::subregion_nloptimizer(
            A_ref_current,
            A_cur,
            roi_current,
            scalefactor,
            interp_type,
            subregion_type,
            r
        );

        auto seed_results = analyze_seeds(
            sr_nloptimizer,
            A_ref_current,
            roi_current,
            seeds_current,
            r,
            50,
            0.1,
            0.5,
            debug
        );

        if (seed_results.success) {
            if (debug) {
                std::cout << "Seed schedule succeeded for frame " << (idx + 1) << std::endl;
            }

            selected_frames.push_back({roi_current, seed_results.seeds, ref_frame_idx_current});
            ++idx;
            continue;
        }

        if (debug) {
            std::cout << "Seed schedule failed for frame " << (idx + 1) << ", updating reference." << std::endl;
        }

        if (idx == 0 || selected_frames.empty()) {
            if (debug) {
                std::cout << "Cannot update reference before the first successful frame; stopping seed schedule." << std::endl;
            }
            break;
        }

        const auto& previous_frame = selected_frames.back();
        if (previous_frame.seed_params_by_region.empty() || region_idx >= static_cast<ROI2D::difference_type>(previous_frame.seed_params_by_region.size())) {
            if (debug) {
                std::cout << "Previous seed frame does not contain region " << region_idx << "; stopping seed schedule." << std::endl;
            }
            break;
        }

        const Array2D<double>& A_prev = A_curs[idx - 1];
        const Array2D<double>& A_prev_ref = previous_frame.ref_frame_idx == 0 ? A_ref : A_curs[previous_frame.ref_frame_idx - 1];
        auto previous_optimizer = details::subregion_nloptimizer(
            A_prev_ref,
            A_prev,
            previous_frame.roi,
            scalefactor,
            interp_type,
            subregion_type,
            r
        );

        auto disps = compute_displacements(
            previous_optimizer,
            previous_frame.roi.reduce(scalefactor),
            previous_frame.seed_params_by_region[region_idx],
            scalefactor,
            cutoff_corrcoef,
            region_idx,
            debug
        );

        auto updated_roi = update(previous_frame.roi, disps, interp_type);
        if (updated_roi.get_points() == 0) {
            if (debug) {
                std::cout << "ROI update removed all points; stopping seed schedule." << std::endl;
            }
            break;
        }

        A_ref_current = A_prev;
        roi_current = std::move(updated_roi);
        seeds_current = propagate_seeds(previous_frame.seed_params_by_region, scalefactor);
        ref_frame_idx_current = static_cast<ROI2D::difference_type>(idx);
    }

    if (debug) {
        std::cout << "Computed seed parameters for " << selected_frames.size() << " frame(s)." << std::endl;
    }

    return selected_frames;
}

std::vector<SeedParams> propagate_seeds(const std::vector<SeedParams>& seeds, ROI2D::difference_type spacing) {
    std::vector<SeedParams> propagated_seeds;
    propagated_seeds.reserve(seeds.size());

    for (const auto& seed : seeds) {
        SeedParams new_seed = seed;
        new_seed.x = seed.x + std::round(seed.u / (spacing + 1)) * (spacing + 1);
        new_seed.y = seed.y + std::round(seed.v / (spacing + 1)) * (spacing + 1);
        propagated_seeds.push_back(new_seed);
    }

    return propagated_seeds;
}

SeedAnalysisResult analyze_seeds(
    details::subregion_nloptimizer &sr_nloptimizer,
    const Array2D<double>& ref_gs,
    const ROI2D& roi,
    const std::vector<SeedParams>& seed_positions,
    ROI2D::difference_type radius,
    int cutoff_iteration,
    double cutoff_max_diffnorm,
    double cutoff_max_corrcoef,
    bool debug
) {
    (void)cutoff_iteration;

    SeedAnalysisResult result;
    result.seeds.reserve(seed_positions.size());
    result.quality.reserve(seed_positions.size());
    result.success = true;

    int region_idx = 0;
    for (const auto& seed_pos : seed_positions) {
        if (seed_pos.x < radius || seed_pos.x >= ref_gs.width() - radius ||
            seed_pos.y < radius || seed_pos.y >= ref_gs.height() - radius) {
            result.success = false;
            ++region_idx;
            continue;
        }

        if (!roi.get_mask()(seed_pos.y, seed_pos.x)) {
            result.success = false;
            ++region_idx;
            continue;
        }

        Array2D<double> params_init(10, 1);
        params_init(0) = seed_pos.y;
        params_init(1) = seed_pos.x;
        params_init(2) = 0.0;
        params_init(3) = 0.0;
        params_init(4) = 0.0;
        params_init(5) = 0.0;
        params_init(6) = 0.0;
        params_init(7) = 0.0;
        params_init(8) = 0.0;
        params_init(9) = 0.0;

        auto result_pair = sr_nloptimizer.global(params_init);
        const auto& params_guess = result_pair.first;

        if (!result_pair.second) {
            if (debug) {
                std::cout << region_idx << ": Seed analysis failed at seed " << seed_pos.x << ", " << seed_pos.y << " => global failed" << std::endl;
            }
            result.success = false;
            ++region_idx;
            continue;
        }

        auto result_iter = sr_nloptimizer(params_guess);
        const auto& params_result = result_iter.first;

        if (!result_iter.second) {
            if (debug) {
                std::cout << region_idx << ": Seed analysis failed at seed " << seed_pos.x << ", " << seed_pos.y << " => iterative search failed" << std::endl;
            }
            result.success = false;
            ++region_idx;
            continue;
        }

        SeedParams seed_result;
        seed_result.x = seed_pos.x;
        seed_result.y = seed_pos.y;
        seed_result.v = params_result(2);
        seed_result.u = params_result(3);
        seed_result.dv_dx = params_result(4);
        seed_result.dv_dy = params_result(5);
        seed_result.du_dx = params_result(6);
        seed_result.du_dy = params_result(7);
        seed_result.corrcoef = params_result(8);

        SeedQuality quality;
        quality.num_iterations = sr_nloptimizer.get_last_iteration_count();
        quality.diffnorm = params_result(9);

        result.seeds.push_back(seed_result);
        result.quality.push_back(quality);

        if (seed_result.corrcoef > cutoff_max_corrcoef || quality.diffnorm > cutoff_max_diffnorm) {
            if (debug) {
                std::cout << region_idx << ": Seed analysis failed at seed " << seed_pos.x << ", " << seed_pos.y
                          << " => corrcoef: " << seed_result.corrcoef
                          << ", diffnorm: " << quality.diffnorm << std::endl;
            }
            result.success = false;
        }
        ++region_idx;
    }

    return result;
}

} // namespace algo

namespace {

using matlab_difference_type = ROI2D::difference_type;

struct matlab_seed_segment final {
    matlab_difference_type ref_idx = 0;
    std::vector<std::vector<SeedParams>> seeds_by_frame;
    std::vector<SeedParams> terminal_seeds;
};

Array2D<double> matlab_make_seed_params_init(const SeedParams& seed) {
    Array2D<double> params_init(10, 1);
    params_init(0) = seed.y;
    params_init(1) = seed.x;
    params_init(2) = 0.0;
    params_init(3) = 0.0;
    params_init(4) = 0.0;
    params_init(5) = 0.0;
    params_init(6) = 0.0;
    params_init(7) = 0.0;
    params_init(8) = 0.0;
    params_init(9) = 0.0;
    return params_init;
}

bool matlab_seed_matches_region(const ROI2D& roi_reduced,
                                const SeedParams& seed,
                                matlab_difference_type scalefactor,
                                matlab_difference_type region_idx) {
    if (scalefactor <= 0) {
        return false;
    }
    const matlab_difference_type reduced_y = seed.y / scalefactor;
    const matlab_difference_type reduced_x = seed.x / scalefactor;
    const auto region_idx_pair = roi_reduced.get_region_idx(reduced_y, reduced_x);
    return region_idx_pair.first == region_idx;
}

bool matlab_seed_positions_are_unique(const std::vector<SeedParams>& seeds,
                                      matlab_difference_type scalefactor) {
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        for (std::size_t j = i + 1; j < seeds.size(); ++j) {
            if (seeds[i].x / scalefactor == seeds[j].x / scalefactor &&
                seeds[i].y / scalefactor == seeds[j].y / scalefactor) {
                return false;
            }
        }
    }
    return true;
}

std::vector<SeedParams> matlab_propagate_seeds(const std::vector<SeedParams>& seeds,
                                               matlab_difference_type scalefactor) {
    std::vector<SeedParams> propagated_seeds;
    propagated_seeds.reserve(seeds.size());

    for (const auto& seed : seeds) {
        SeedParams propagated = seed;
        propagated.x = seed.x + static_cast<matlab_difference_type>(std::round(seed.u / scalefactor) * scalefactor);
        propagated.y = seed.y + static_cast<matlab_difference_type>(std::round(seed.v / scalefactor) * scalefactor);
        propagated.u = 0.0;
        propagated.v = 0.0;
        propagated.du_dx = 0.0;
        propagated.du_dy = 0.0;
        propagated.dv_dx = 0.0;
        propagated.dv_dy = 0.0;
        propagated.corrcoef = 0.0;
        propagated_seeds.push_back(propagated);
    }

    return propagated_seeds;
}

matlab_seed_segment matlab_compute_seed_segment(const DIC_analysis_parallel_input& input,
                                                matlab_difference_type ref_idx,
                                                const ROI2D& roi_ref,
                                                const std::vector<SeedParams>& seeds_by_region,
                                                bool seeds_are_optimized) {
    typedef ROI2D::difference_type difference_type;

    const auto& DIC_input = input.base_input;
    matlab_seed_segment segment;
    segment.ref_idx = ref_idx;

    if (seeds_by_region.empty()) {
        return segment;
    }

    auto roi_reduced = roi_ref.reduce(DIC_input.scalefactor);
    if (difference_type(seeds_by_region.size()) != roi_reduced.size_regions()) {
        throw std::invalid_argument("matlab_DIC_analysis requires seeds_by_region.size() to match the number of ROI regions.");
    }

    std::vector<SeedParams> predicted_seeds = seeds_are_optimized ?
        matlab_propagate_seeds(seeds_by_region, DIC_input.scalefactor) :
        seeds_by_region;

    if (!matlab_seed_positions_are_unique(predicted_seeds, DIC_input.scalefactor)) {
        throw std::invalid_argument("matlab_DIC_analysis received duplicate seed positions on the reduced grid.");
    }

    segment.seeds_by_frame.reserve(DIC_input.imgs.size() - ref_idx - 1);

    const auto A_ref = DIC_input.imgs[ref_idx].get_gs();
    for (difference_type cur_idx = ref_idx + 1; cur_idx < difference_type(DIC_input.imgs.size()); ++cur_idx) {
        const auto A_cur = DIC_input.imgs[cur_idx].get_gs();
        auto sr_nloptimizer = details::subregion_nloptimizer(
            A_ref,
            A_cur,
            roi_ref,
            DIC_input.scalefactor,
            DIC_input.interp_type,
            DIC_input.subregion_type,
            DIC_input.r
        );

        std::vector<SeedParams> optimized_seeds;
        optimized_seeds.reserve(predicted_seeds.size());
        bool frame_success = matlab_seed_positions_are_unique(predicted_seeds, DIC_input.scalefactor);

        for (difference_type region_idx = 0;
             frame_success && region_idx < difference_type(predicted_seeds.size());
             ++region_idx) {
            const auto& predicted_seed = predicted_seeds[region_idx];
            if (!matlab_seed_matches_region(roi_reduced, predicted_seed, DIC_input.scalefactor, region_idx)) {
                if (DIC_input.debug) {
                    std::cout << "matlab_DIC_analysis: seed for region " << region_idx
                              << " is outside its ROI at frame " << cur_idx << "." << std::endl;
                }
                frame_success = false;
                break;
            }

            auto params_init = matlab_make_seed_params_init(predicted_seed);
            auto global_result = sr_nloptimizer.global(params_init);
            if (!global_result.second) {
                if (DIC_input.debug) {
                    std::cout << "matlab_DIC_analysis: global seed optimization failed for region "
                              << region_idx << " at frame " << cur_idx << "." << std::endl;
                }
                frame_success = false;
                break;
            }

            SeedParams optimized_seed = SeedParams::from_array(global_result.first);
            const double diffnorm = global_result.first(9);
            const int num_iterations = sr_nloptimizer.get_last_iteration_count();
            const bool iteration_saturated = (cur_idx > ref_idx + 1) && (num_iterations >= 100);

            if (optimized_seed.corrcoef > input.cutoff_max_corrcoef ||
                diffnorm > input.cutoff_max_diffnorm ||
                iteration_saturated ||
                !matlab_seed_matches_region(roi_reduced, optimized_seed, DIC_input.scalefactor, region_idx)) {
                if (DIC_input.debug) {
                    std::cout << "matlab_DIC_analysis: seed quality failed for region " << region_idx
                              << " at frame " << cur_idx
                              << " corrcoef=" << optimized_seed.corrcoef
                              << " diffnorm=" << diffnorm
                              << " iterations=" << num_iterations
                              << " saturated=" << (iteration_saturated ? "yes" : "no") << "." << std::endl;
                }
                frame_success = false;
                break;
            }

            optimized_seeds.push_back(optimized_seed);
        }

        if (!frame_success || !matlab_seed_positions_are_unique(optimized_seeds, DIC_input.scalefactor)) {
            if (DIC_input.debug) {
                std::cout << "matlab_DIC_analysis: stopping seed schedule at frame " << cur_idx << "." << std::endl;
            }
            break;
        }

        segment.seeds_by_frame.push_back(optimized_seeds);
    }

    if (!segment.seeds_by_frame.empty()) {
        segment.terminal_seeds = segment.seeds_by_frame.back();
    }

    return segment;
}

std::vector<Disp2D> matlab_run_segment_dic(const DIC_analysis_parallel_input& input,
                                           const ROI2D& roi_ref,
                                           const matlab_seed_segment& segment,
                                           bool run_in_parallel) {
    typedef ROI2D::difference_type difference_type;

    std::vector<Disp2D> segment_disps(segment.seeds_by_frame.size());
    if (segment.seeds_by_frame.empty()) {
        return segment_disps;
    }

    const auto& DIC_input = input.base_input;
    const auto A_ref = DIC_input.imgs[segment.ref_idx].get_gs();
    const difference_type num_frames = static_cast<difference_type>(segment.seeds_by_frame.size());

    auto compute_frame = [&](difference_type frame_idx) {
        const difference_type cur_idx = segment.ref_idx + frame_idx + 1;
        const auto A_cur = DIC_input.imgs[cur_idx].get_gs();
        return RGDIC_without_thread(
            A_ref,
            A_cur,
            roi_ref,
            DIC_input.scalefactor,
            DIC_input.interp_type,
            DIC_input.subregion_type,
            DIC_input.r,
            DIC_input.cutoff_corrcoef,
            DIC_input.debug,
            segment.seeds_by_frame[frame_idx],
            true
        );
    };

    if (!run_in_parallel || num_frames <= 1) {
        for (difference_type frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            segment_disps[frame_idx] = compute_frame(frame_idx);
        }
        return segment_disps;
    }

    #pragma omp parallel for num_threads(std::min(num_frames, DIC_input.num_threads)) schedule(dynamic)
    for (difference_type frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        segment_disps[frame_idx] = compute_frame(frame_idx);
    }

    return segment_disps;
}

DIC_analysis_output matlab_DIC_analysis_impl(const DIC_analysis_parallel_input& input,
                                             bool run_in_parallel,
                                             const std::string& mode_name) {
    typedef ROI2D::difference_type difference_type;

    const auto& DIC_input = input.base_input;
    if (DIC_input.imgs.size() < 2) {
        throw std::invalid_argument("matlab_DIC_analysis requires at least 2 images.");
    }
    if (input.seeds_by_region.empty()) {
        throw std::invalid_argument("matlab_DIC_analysis requires one manual/preset seed per ROI region.");
    }

    DIC_analysis_output DIC_output;
    DIC_output.disps.resize(DIC_input.imgs.size() - 1);
    DIC_output.perspective_type = PERSPECTIVE::LAGRANGIAN;
    DIC_output.units = "pixels";
    DIC_output.units_per_pixel = 1.0;

    std::vector<Disp2D> step_disps;
    std::vector<ROI2D> step_rois;
    std::vector<difference_type> step_ref_idx;
    if (DIC_input.save_disps_steps) {
        step_disps.resize(DIC_output.disps.size());
        step_rois.resize(DIC_output.disps.size());
        step_ref_idx.resize(DIC_output.disps.size());
    }

    difference_type ref_idx = 0;
    ROI2D roi_ref = DIC_input.roi;
    std::vector<SeedParams> current_seeds = input.seeds_by_region;
    bool current_seeds_optimized = input.seeds_are_optimized;

    while (ref_idx < difference_type(DIC_input.imgs.size()) - 1) {
        const auto segment = matlab_compute_seed_segment(input, ref_idx, roi_ref, current_seeds, current_seeds_optimized);

        if (segment.seeds_by_frame.empty()) {
            if (ref_idx == 0) {
                throw std::runtime_error(mode_name + " could not seed any current image.");
            }
            throw std::runtime_error(
                mode_name + " could not seed the segment starting at reference frame " +
                std::to_string(ref_idx + 1) + "."
            );
        }

        const auto segment_disps = matlab_run_segment_dic(input, roi_ref, segment, run_in_parallel);
        ROI2D next_roi_ref;
        for (difference_type frame_idx = 0; frame_idx < static_cast<difference_type>(segment_disps.size()); ++frame_idx) {
            const difference_type cur_idx = ref_idx + frame_idx + 1;
            DIC_output.disps[cur_idx - 1] = segment_disps[frame_idx];
            next_roi_ref = matlab_update_roi(roi_ref, segment_disps[frame_idx], DIC_input.interp_type, DIC_input.r);

            if (DIC_input.save_disps_steps) {
                step_disps[cur_idx - 1] = segment_disps[frame_idx];
                step_rois[cur_idx - 1] = roi_ref;
                step_ref_idx[cur_idx - 1] = ref_idx;
            }
        }

        const difference_type segment_end_idx = ref_idx + static_cast<difference_type>(segment.seeds_by_frame.size());
        if (DIC_input.debug) {
            std::cout << "matlab_DIC_analysis: processed segment "
                      << (ref_idx + 1) << " -> " << (segment_end_idx + 1)
                      << " (" << segment.seeds_by_frame.size() << " frame(s))." << std::endl;
        }

        ref_idx = segment_end_idx;
        if (ref_idx >= difference_type(DIC_input.imgs.size()) - 1) {
            break;
        }

        roi_ref = next_roi_ref;
        current_seeds = segment.terminal_seeds;
        current_seeds_optimized = true;
    }

    if (DIC_input.save_disps_steps && !step_disps.empty()) {
        DIC_analysis_step_data step_data;
        step_data.step_disps = step_disps;
        step_data.step_rois = step_rois;
        step_data.step_ref_idx = step_ref_idx;

        const std::string step_filename = run_in_parallel ?
            "matlab_DIC_analysis_parallel_step_data.bin" :
            "matlab_DIC_analysis_sequential_step_data.bin";
        save(step_data, step_filename);
        if (DIC_input.debug) {
            std::cout << "Step displacement data saved to " << step_filename << std::endl;
        }
    }

    return DIC_output;
}

} // namespace

DIC_analysis_output matlab_DIC_analysis_sequential(const DIC_analysis_input &DIC_input,
                                                   const std::vector<SeedParams>& seeds_by_region,
                                                   bool seeds_are_optimized) {
    return matlab_DIC_analysis_sequential(DIC_analysis_parallel_input(DIC_input, seeds_by_region, seeds_are_optimized));
}

DIC_analysis_output matlab_DIC_analysis_sequential(const DIC_analysis_parallel_input &input) {
    return matlab_DIC_analysis_impl(input, false, "matlab_DIC_analysis_sequential");
}

DIC_analysis_output matlab_DIC_analysis_parallel(const DIC_analysis_parallel_input& input) {
    return matlab_DIC_analysis_impl(input, true, "matlab_DIC_analysis_parallel");
}

DIC_analysis_output DIC_analysis_parallel(const DIC_analysis_parallel_input& input) {
    typedef std::ptrdiff_t difference_type;

    const auto& DIC_input = input.base_input;
    if (DIC_input.imgs.size() < 2) {
        throw std::invalid_argument("DIC_analysis_parallel requires at least 2 images.");
    }
    if (input.seeds_by_region.empty()) {
        throw std::invalid_argument("DIC_analysis_parallel requires seeds_by_region to be provided.");
    }

    if (DIC_input.debug) {
        std::cout << "\nStarting seed-based parallel DIC analysis..." << std::endl;
        std::cout << "Number of regions: " << input.seeds_by_region.size() << std::endl;
        std::cout << "Cutoff max diffnorm: " << input.cutoff_max_diffnorm << std::endl;
        std::cout << "Cutoff max corrcoef: " << input.cutoff_max_corrcoef << std::endl;
    }

    const auto start_analysis = std::chrono::system_clock::now();

    DIC_analysis_output DIC_output;
    DIC_output.disps.resize(DIC_input.imgs.size() - 1);
    DIC_output.perspective_type = PERSPECTIVE::LAGRANGIAN;
    DIC_output.units = "pixels";
    DIC_output.units_per_pixel = 1.0;

    std::vector<Array2D<double>> A_curs;
    A_curs.reserve(DIC_input.imgs.size() - 1);
    for (std::size_t i = 1; i < DIC_input.imgs.size(); ++i) {
        A_curs.push_back(DIC_input.imgs[i].get_gs());
    }

    auto seed_schedule = algo::compute_only_seed_points(
        DIC_input.imgs[0].get_gs(),
        A_curs,
        DIC_input.roi,
        DIC_input.scalefactor,
        DIC_input.interp_type,
        DIC_input.subregion_type,
        DIC_input.r,
        input.seeds_by_region,
        DIC_input.cutoff_corrcoef,
        0,
        DIC_input.debug
    );

    if (seed_schedule.empty()) {
        throw std::runtime_error("No valid seed parameters could be computed.");
    }

    const std::size_t safe_batch_size = seed_schedule.size();
    if (DIC_input.debug) {
        std::cout << "Successfully computed seed parameters for " << safe_batch_size << " frame(s)." << std::endl;
    }

    #pragma omp parallel for num_threads(std::min(static_cast<difference_type>(safe_batch_size), DIC_input.num_threads)) schedule(dynamic)
    for (std::size_t i = 0; i < safe_batch_size; ++i) {
        const difference_type frame_idx = static_cast<difference_type>(i) + 1;
        const auto& frame_data = seed_schedule[i];

        if (frame_data.seed_params_by_region.empty()) {
            if (DIC_input.debug) {
                #pragma omp critical
                {
                    std::cerr << "Skipping frame " << frame_idx << " because no seed parameters were available." << std::endl;
                }
            }
            continue;
        }

        try {
            auto sr_nloptimizer = details::subregion_nloptimizer(
                DIC_input.imgs[frame_data.ref_frame_idx].get_gs(),
                DIC_input.imgs[frame_idx].get_gs(),
                frame_data.roi,
                DIC_input.scalefactor,
                DIC_input.interp_type,
                DIC_input.subregion_type,
                DIC_input.r
            );

            auto disps = algo::compute_displacements(
                sr_nloptimizer,
                frame_data.roi.reduce(DIC_input.scalefactor),
                frame_data.seed_params_by_region.front(),
                DIC_input.scalefactor,
                DIC_input.cutoff_corrcoef,
                0,
                DIC_input.debug
            );

            DIC_output.disps[frame_idx - 1] = std::move(disps);
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Frame " << frame_idx << " failed during parallel seed-based DIC: " << e.what() << std::endl;
            }
        }
    }

    if (DIC_input.debug) {
        const auto end_analysis = std::chrono::system_clock::now();
        const std::chrono::duration<double> elapsed_seconds = end_analysis - start_analysis;
        std::cout << "Total parallel DIC analysis time: " << elapsed_seconds.count() << " seconds" << std::endl;
    }

    return DIC_output;
}

} // namespace ncorr
