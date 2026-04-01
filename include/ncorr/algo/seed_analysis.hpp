#ifndef NCORR_ALGO_SEED_ANALYSIS_HPP
#define NCORR_ALGO_SEED_ANALYSIS_HPP

#include "../../ncorr.h"

namespace ncorr {
namespace algo {

struct SeedQuality final {
    int num_iterations = 0;
    double diffnorm = 0.0;
};

struct SeedAnalysisResult final {
    std::vector<SeedParams> seeds;
    std::vector<SeedQuality> quality;
    bool success = false;
};

struct SeedScheduleFrame final {
    ROI2D roi;
    std::vector<SeedParams> seed_params_by_region;
    ROI2D::difference_type ref_frame_idx = 0;
};

SeedAnalysisResult analyze_seeds(
    details::subregion_nloptimizer &sr_nloptimizer,
    const Array2D<double>& ref_gs,
    const ROI2D& roi,
    const std::vector<SeedParams>& seed_positions,
    ROI2D::difference_type radius,
    int cutoff_iteration,
    double cutoff_max_diffnorm,
    double cutoff_max_corrcoef,
    bool debug = true
);

Disp2D compute_displacements(
    details::subregion_nloptimizer &sr_nloptimizer,
    const ROI2D& roi_reduced,
    const SeedParams& seedparams,
    ROI2D::difference_type scalefactor,
    double cutoff_corrcoef,
    ROI2D::difference_type region_idx,
    bool debug
);

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
    ROI2D::difference_type region_idx = 0,
    bool debug = false
);

std::vector<SeedParams> propagate_seeds(
    const std::vector<SeedParams>& seeds,
    ROI2D::difference_type spacing
);

} // namespace algo
} // namespace ncorr

#endif
