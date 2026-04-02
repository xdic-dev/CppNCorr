/* 
 * File:   Array2D.h
 * Author: justin
 *
 * Created on December 30, 2014, 4:55 PM
 */

#include "Array2DFFTW.h"

extern "C" {
    void dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
}

namespace ncorr {

namespace {

enum class FFTWOperation { Conv, Deconv, Xcorr };

void check_kernel_size(const Array2D<double> &A, const Array2D<double> &kernel) {
    if (!(kernel.height() % 2) || !(kernel.width() % 2) ||
        kernel.width() > A.width() || kernel.height() > A.height()) {
        throw std::invalid_argument("Attempted to convolve matrix of size " + A.size_2D_string() +
                                    " with kernel of size " + kernel.size_2D_string() +
                                    ". Kernel must have odd dimensions and be equal to or smaller than matrix.");
    }
}

Array2D<double> conv_base(const Array2D<double> &A,
                          const Array2D<double> &kernel,
                          FFTWOperation fftw_type) {
    check_kernel_size(A, kernel);

    std::unique_lock<std::mutex> fftw_lock(details::fftw_mutex);

    const Array2D<double>::difference_type H = A.height();
    const Array2D<double>::difference_type W = A.width();
    const Array2D<double>::difference_type halfWplus1 = (W / 2) + 1;

    double* A_rm = static_cast<double*>(fftw_malloc(sizeof(double) * H * W));
    double* K_rm = static_cast<double*>(fftw_malloc(sizeof(double) * H * W));
    if (!A_rm || !K_rm) {
        if (A_rm) fftw_free(A_rm);
        if (K_rm) fftw_free(K_rm);
        throw std::bad_alloc();
    }

    std::fill(A_rm, A_rm + H * W, 0.0);
    std::fill(K_rm, K_rm + H * W, 0.0);

    for (Array2D<double>::difference_type r = 0; r < H; ++r) {
        for (Array2D<double>::difference_type c = 0; c < W; ++c) {
            A_rm[r * W + c] = A(r, c);
        }
    }

    const Array2D<double>::difference_type KH = kernel.height();
    const Array2D<double>::difference_type KW = kernel.width();
    const Array2D<double>::difference_type c_kh = (KH - 1) / 2;
    const Array2D<double>::difference_type c_kw = (KW - 1) / 2;

    for (Array2D<double>::difference_type r = 0; r <= c_kh; ++r) {
        for (Array2D<double>::difference_type c = 0; c <= c_kw; ++c) {
            K_rm[r * W + c] = kernel(c_kh + r, c_kw + c);
        }
    }
    if (c_kw > 0) {
        for (Array2D<double>::difference_type r = 0; r <= c_kh; ++r) {
            for (Array2D<double>::difference_type c = 0; c < c_kw; ++c) {
                K_rm[r * W + (W - c_kw + c)] = kernel(c_kh + r, c);
            }
        }
    }
    if (c_kh > 0) {
        for (Array2D<double>::difference_type r = 0; r < c_kh; ++r) {
            for (Array2D<double>::difference_type c = 0; c <= c_kw; ++c) {
                K_rm[(H - c_kh + r) * W + c] = kernel(r, c_kh + c);
            }
        }
    }
    if (c_kh > 0 && c_kw > 0) {
        for (Array2D<double>::difference_type r = 0; r < c_kh; ++r) {
            for (Array2D<double>::difference_type c = 0; c < c_kw; ++c) {
                K_rm[(H - c_kh + r) * W + (W - c_kw + c)] = kernel(r, c);
            }
        }
    }

    fftw_complex* A_fft = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * H * halfWplus1));
    fftw_complex* K_fft = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * H * halfWplus1));
    if (!A_fft || !K_fft) {
        if (A_fft) fftw_free(A_fft);
        if (K_fft) fftw_free(K_fft);
        fftw_free(A_rm);
        fftw_free(K_rm);
        throw std::bad_alloc();
    }

    fftw_plan plan_A = fftw_plan_dft_r2c_2d(static_cast<int>(H), static_cast<int>(W), A_rm, A_fft, FFTW_ESTIMATE);
    fftw_plan plan_kernel = fftw_plan_dft_r2c_2d(static_cast<int>(H), static_cast<int>(W), K_rm, K_fft, FFTW_ESTIMATE);
    fftw_plan plan_output = fftw_plan_dft_c2r_2d(static_cast<int>(H), static_cast<int>(W), A_fft, A_rm, FFTW_ESTIMATE);

    auto plan_deleter = [](fftw_plan *p) { return fftw_destroy_plan(*p); };
    std::unique_ptr<fftw_plan, decltype(plan_deleter)> plan_A_delete(&plan_A, plan_deleter);
    std::unique_ptr<fftw_plan, decltype(plan_deleter)> plan_kernel_delete(&plan_kernel, plan_deleter);
    std::unique_ptr<fftw_plan, decltype(plan_deleter)> plan_output_delete(&plan_output, plan_deleter);

    fftw_lock.unlock();

    fftw_execute(plan_A);
    fftw_execute(plan_kernel);

    for (Array2D<double>::difference_type r = 0; r < H; ++r) {
        const Array2D<double>::difference_type row_off = r * halfWplus1;
        for (Array2D<double>::difference_type c = 0; c < halfWplus1; ++c) {
            const Array2D<double>::difference_type idx = row_off + c;
            const double ar = A_fft[idx][0];
            const double ai = A_fft[idx][1];
            const double kr = K_fft[idx][0];
            const double ki = K_fft[idx][1];

            switch (fftw_type) {
                case FFTWOperation::Conv:
                    A_fft[idx][0] = ar * kr - ai * ki;
                    A_fft[idx][1] = ar * ki + ai * kr;
                    break;
                case FFTWOperation::Deconv: {
                    const double den = kr * kr + ki * ki;
                    if (den == 0.0) {
                        A_fft[idx][0] = 0.0;
                        A_fft[idx][1] = 0.0;
                    } else {
                        A_fft[idx][0] = (ar * kr + ai * ki) / den;
                        A_fft[idx][1] = (ai * kr - ar * ki) / den;
                    }
                    break;
                }
                case FFTWOperation::Xcorr:
                    A_fft[idx][0] = ar * kr + ai * ki;
                    A_fft[idx][1] = -ar * ki + ai * kr;
                    break;
            }
        }
    }

    fftw_execute(plan_output);
    fftw_lock.lock();

    Array2D<double> C(H, W);
    const double scale = 1.0 / static_cast<double>(H * W);
    for (Array2D<double>::difference_type r = 0; r < H; ++r) {
        for (Array2D<double>::difference_type c = 0; c < W; ++c) {
            C(r, c) = A_rm[r * W + c] * scale;
        }
    }

    fftw_free(A_rm);
    fftw_free(K_rm);
    fftw_free(A_fft);
    fftw_free(K_fft);

    return C;
}

} // namespace
    
namespace details {     
    // The only thread-safe routine in FFTW is fftw_execute(), so use this mutex
    // for all other routines.
    std::mutex fftw_mutex;    

    Array2D<double> blas_mat_mult(const Array2D<double> &A, const Array2D<double> &B) {
        if (A.width() != B.height()) {
            throw std::invalid_argument("Attempted to multiply matrix of size " + A.size_2D_string() + " with " +
                                        B.size_2D_string() + ". These sizes are incompatible for matrix multiplication.");
        }

        Array2D<double> C(A.height(), B.width());

        char TRANS = 'N';
        int M = A.height();
        int N = B.width();
        int K = A.width();
        double ALPHA = 1.0;
        double BETA = 0.0;
        dgemm_(&TRANS, &TRANS, &M, &N, &K, &ALPHA, A.get_pointer(), &M, B.get_pointer(), &K, &BETA, C.get_pointer(), &M);

        return C;
    }
}

Array2D<double> conv(const Array2D<double> &A, const Array2D<double> &B) {
    return conv_base(A, B, FFTWOperation::Conv);
}

Array2D<double> deconv(const Array2D<double> &A, const Array2D<double> &B) {
    return conv_base(A, B, FFTWOperation::Deconv);
}

Array2D<double> xcorr(const Array2D<double> &A, const Array2D<double> &B) {
    return conv_base(A, B, FFTWOperation::Xcorr);
}
    
}
