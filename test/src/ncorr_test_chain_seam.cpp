// Regression test for the multi-reference chain composition seam.
//
// Covers two layers, in order from low-level to high-level:
//
//   1) Array2D::quintic_interp round-trip at integer query points.
//      A correct biquintic B-spline interpolator MUST return the source
//      array value at every integer (p1, p2) within the array's interior.
//      Before the Unser recursive bcoef fix in Array2D.h, the FFT-based
//      circular deconv in get_bspline_mat_ptr returned values inflated by
//      ~+29% for any array whose top<->bottom or left<->right edges differed
//      (the circular wrap created a synthetic discontinuity that the
//      deconvolution amplified into a spectral bias). This caused the
//      seam-of-segment chain composition in matlab_DIC_analysis_* to
//      mis-evaluate the bridge field by ~45 px at the first segment
//      boundary even though source data was correct.
//
//   2) exact_add_with_rois chain composition on a synthetic 2-link chain.
//      Build a constant bridge disp d_b = (V_b, U_b) over a small rectangular
//      ROI A, and a constant target disp d_t = (V_t, U_t) over the warped
//      ROI A_w = A + (V_b, U_b)/scalefactor. The composed chain result at
//      every interior point of A must be exactly (V_b + V_t, U_b + U_t).
//      With the seam bug, the result over-shot by ~+29% of the bridge
//      magnitude; with the fix it round-trips within ~1e-6.
//
// Run: ./ncorr_test_chain_seam
// Exit: 0 on success, 1 on any failure.

#include "ncorr.h"
#include "Array2D.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ncorr;

namespace {

int g_failures = 0;

void check_close(const std::string& label,
                 double got, double expected, double tol = 1e-6) {
    const double diff = std::abs(got - expected);
    if (diff > tol) {
        std::cerr << "  FAIL: " << label
                  << "  got=" << std::setprecision(10) << got
                  << "  expected=" << std::setprecision(10) << expected
                  << "  |diff|=" << diff
                  << "  tol=" << tol << std::endl;
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Test 1: Array2D::quintic_interp must round-trip at integer query points
// for arrays with non-trivial gradients (was broken by FFT circular deconv).
// ---------------------------------------------------------------------------
void test_quintic_interp_roundtrip() {
    std::cout << "[test 1] Array2D::quintic_interp round-trip at integer queries\n";

    struct Case {
        std::string name;
        std::function<double(int, int)> f;
    };
    const std::vector<Case> cases = {
        {"constant 188",        [](int, int)         { return 188.0; }},
        {"row-linear 0..200",   [](int r, int)       { return double(r) * 4.0; }},
        {"col-linear 0..200",   [](int, int c)       { return double(c) * 4.0; }},
        {"diagonal gradient",   [](int r, int c)     { return double(r + c) * 2.0; }},
        {"steep row gradient",  [](int r, int)       { return double(r) * 50.0; }},  // 0..2500 across 50 rows
        {"sinusoid",            [](int r, int c)     {
                                    return 100.0 + 80.0 * std::sin(0.4 * r) * std::cos(0.3 * c);
                                }},
    };

    constexpr int H = 50, W = 60;
    for (const auto& tc : cases) {
        Array2D<double> A(H, W);
        for (int p2 = 0; p2 < W; ++p2) {
            for (int p1 = 0; p1 < H; ++p1) {
                A(p1, p2) = tc.f(p1, p2);
            }
        }
        auto interp = A.get_interpolator(INTERP::QUINTIC_BSPLINE);
        double max_err = 0.0;
        int worst_p1 = 0, worst_p2 = 0;
        // Test integer queries strictly inside the array (avoid the very
        // outermost cells where any reasonable B-spline impl can have
        // boundary-dependent values).
        for (int p2 = 2; p2 < W - 2; ++p2) {
            for (int p1 = 2; p1 < H - 2; ++p1) {
                const double got = interp(double(p1), double(p2));
                const double err = std::abs(got - A(p1, p2));
                if (err > max_err) { max_err = err; worst_p1 = p1; worst_p2 = p2; }
            }
        }
        const double tol = 1e-6;
        const std::string label = "    " + tc.name +
            "  max|interp-src|=" + std::to_string(max_err) +
            " at (" + std::to_string(worst_p1) + "," + std::to_string(worst_p2) + ")";
        if (max_err > tol) {
            std::cerr << "  FAIL: " << label << "  tol=" << tol << std::endl;
            ++g_failures;
        } else {
            std::cout << "    PASS: " << tc.name
                      << "  max|interp-src|=" << max_err << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: exact_add_with_rois composes two constant-displacement links into
// (V_b + V_t, U_b + U_t) at every interior point of the first ROI.
// ---------------------------------------------------------------------------
//
// We construct:
//   - A small rectangular reduced-grid ROI A of size H_red x W_red placed
//     at (top, left) in a full-canvas of size canvas_h x canvas_w (reduced).
//   - A bridge Disp2D with constant (V_b, U_b) stored at every cell of A.
//   - A warped ROI A_w at (top + dy, left + dx) where (dy, dx) =
//     round(V_b/sf), round(U_b/sf).
//   - A target Disp2D with constant (V_t, U_t) stored at every cell of A_w.
//
// exact_add_with_rois({bridge, target}, {bridge.roi(), target.roi()}) must
// produce, at every interior cell of A, the value (V_b + V_t, U_b + U_t).

namespace {

Disp2D build_constant_disp(ROI2D::difference_type canvas_h,
                           ROI2D::difference_type canvas_w,
                           ROI2D::difference_type top,
                           ROI2D::difference_type bot,
                           ROI2D::difference_type left,
                           ROI2D::difference_type right,
                           double V, double U,
                           ROI2D::difference_type scalefactor) {
    using D = ROI2D::difference_type;
    Array2D<bool>   mask(canvas_h, canvas_w);
    Array2D<double> v_arr(canvas_h, canvas_w);
    Array2D<double> u_arr(canvas_h, canvas_w);
    Array2D<double> cc_arr(canvas_h, canvas_w);
    cc_arr() = 1.0;
    for (D p2 = left; p2 <= right; ++p2) {
        for (D p1 = top; p1 <= bot; ++p1) {
            mask (p1, p2) = true;
            v_arr(p1, p2) = V;
            u_arr(p1, p2) = U;
            cc_arr(p1, p2) = 0.5;
        }
    }
    ROI2D roi(std::move(mask));
    return Disp2D(std::move(v_arr), std::move(u_arr), std::move(cc_arr),
                  std::move(roi), scalefactor);
}

} // namespace

void test_chain_composition_2link_constant() {
    std::cout << "[test 2] exact_add_with_rois: 2-link constant-disp chain\n";

    using D = ROI2D::difference_type;
    const D canvas_h = 200;
    const D canvas_w = 200;
    const D scalefactor = 11;  // matches the production case

    // Bridge ROI placed in upper-left quadrant.
    const D b_top = 30,  b_bot = 82,  b_left = 31,  b_right = 125;
    const D b_H = b_bot - b_top + 1;
    const D b_W = b_right - b_left + 1;

    // The four scenarios below exercise the regimes that previously broke:
    // small disp, large disp, both signs.
    struct Case {
        std::string name;
        double V_b, U_b;
        double V_t, U_t;
    };
    const std::vector<Case> cases = {
        {"small bridge + small target",   1.0,    0.5,    0.2,   0.1},
        {"large bridge + small target", 155.5,  36.5,    0.66,  0.16},  // production case
        {"large bridge negative",      -170.0, -55.0,    0.40,  0.20},
        {"asymmetric",                  188.0,  22.5,    1.05,  0.35},
    };

    for (const auto& tc : cases) {
        // Build bridge disp (over original ROI A).
        Disp2D bridge = build_constant_disp(
            canvas_h, canvas_w,
            b_top, b_bot, b_left, b_right,
            tc.V_b, tc.U_b, scalefactor);

        // Warp the target ROI by (V_b/sf, U_b/sf) in reduced units.
        const D dy = static_cast<D>(std::round(tc.V_b / double(scalefactor)));
        const D dx = static_cast<D>(std::round(tc.U_b / double(scalefactor)));
        const D t_top  = b_top  + dy;
        const D t_bot  = b_bot  + dy;
        const D t_left = b_left + dx;
        const D t_right= b_right+ dx;
        if (t_top < 0 || t_bot >= canvas_h || t_left < 0 || t_right >= canvas_w) {
            std::cerr << "  FAIL: '" << tc.name << "': warped ROI out of canvas\n";
            ++g_failures;
            continue;
        }
        Disp2D target = build_constant_disp(
            canvas_h, canvas_w,
            t_top, t_bot, t_left, t_right,
            tc.V_t, tc.U_t, scalefactor);

        // Run chain composition.
        auto combined = exact_add_with_rois({bridge, target},
                                            {bridge.get_roi(), target.get_roi()});

        if (combined.get_roi().get_points() == 0) {
            std::cerr << "  FAIL: '" << tc.name << "': empty result ROI\n";
            ++g_failures;
            continue;
        }

        // Verify every valid cell of the OUTPUT equals (V_b + V_t, U_b + U_t).
        // Strict interior check: skip a 2-cell ring near the warped boundary
        // since the extrapolation zone there can include a partial overshoot
        // for any biquintic; the residual seam diagnostic only ever looked
        // at INTERIOR pixels too.
        const auto& v_out = combined.get_v().get_array();
        const auto& u_out = combined.get_u().get_array();
        const D interior_margin = 2;
        const D in_top   = b_top   + interior_margin;
        const D in_bot   = b_bot   - interior_margin;
        const D in_left  = b_left  + interior_margin;
        const D in_right = b_right - interior_margin;

        double max_dv = 0.0, max_du = 0.0;
        D worst_p1_v = 0, worst_p2_v = 0;
        D worst_p1_u = 0, worst_p2_u = 0;
        D n_checked = 0;
        for (D p2 = in_left; p2 <= in_right; ++p2) {
            for (D p1 = in_top; p1 <= in_bot; ++p1) {
                ++n_checked;
                const double v = v_out(p1, p2);
                const double u = u_out(p1, p2);
                const double dv = std::abs(v - (tc.V_b + tc.V_t));
                const double du = std::abs(u - (tc.U_b + tc.U_t));
                if (dv > max_dv) { max_dv = dv; worst_p1_v = p1; worst_p2_v = p2; }
                if (du > max_du) { max_du = du; worst_p1_u = p1; worst_p2_u = p2; }
            }
        }
        // Tolerance: 1e-3 px. For constant fields the bcoef + interp pipeline
        // is exact up to floating-point round-off (~1e-12), but allow generous
        // slack for any harmless smoothing in the chain walk near the seam.
        const double tol = 1e-3;
        if (max_dv > tol || max_du > tol) {
            std::cerr << "  FAIL: '" << tc.name << "'  n=" << n_checked
                      << "  max|dv|=" << max_dv
                      << " at (" << worst_p1_v << "," << worst_p2_v << ")"
                      << "  max|du|=" << max_du
                      << " at (" << worst_p1_u << "," << worst_p2_u << ")"
                      << "  expected=(V_b+V_t,U_b+U_t)=("
                      << (tc.V_b + tc.V_t) << "," << (tc.U_b + tc.U_t) << ")"
                      << "  tol=" << tol << std::endl;
            ++g_failures;
        } else {
            std::cout << "    PASS: '" << tc.name << "'  n=" << n_checked
                      << "  max|dv|=" << max_dv
                      << "  max|du|=" << max_du << std::endl;
        }
    }
    (void)b_H; (void)b_W;
}

} // namespace

int main() {
    std::cout << "===== chain-seam regression test =====" << std::endl;
    test_quintic_interp_roundtrip();
    test_chain_composition_2link_constant();
    std::cout << "======================================" << std::endl;
    if (g_failures == 0) {
        std::cout << "ALL PASS" << std::endl;
        return 0;
    }
    std::cout << g_failures << " failure(s)" << std::endl;
    return 1;
}
