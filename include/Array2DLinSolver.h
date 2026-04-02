#ifndef ARRAY2D_LINSOLVER_H
#define ARRAY2D_LINSOLVER_H

namespace ncorr {

template <typename T, typename T_alloc>
template <typename T_output>
typename std::enable_if<std::is_floating_point<T>::value, T_output>::type
Array2D<T, T_alloc>::get_linsolver(LINSOLVER linsolver_type) const {
    linsolver linsolve;
    switch (linsolver_type) {
        case LINSOLVER::LU:
            linsolve = linsolver(new details::LU_linsolver<Array2D>(*this));
            break;
        case LINSOLVER::QR:
            linsolve = linsolver(new details::QR_linsolver<Array2D>(*this));
            break;
        case LINSOLVER::CHOL:
            linsolve = linsolver(new details::CHOL_linsolver<Array2D>(*this));
            break;
    }

    return linsolve;
}

template <typename T, typename T_alloc>
template <typename T_output>
typename std::enable_if<std::is_floating_point<T>::value, T_output>::type
Array2D<T, T_alloc>::this_linsolve(const Array2D &b) const {
    auto linsolver = ((h == w) ? get_linsolver(LINSOLVER::LU) : get_linsolver(LINSOLVER::QR));

    container x(w, b.width());
    for (difference_type p2 = 0; p2 < b.width(); ++p2) {
        x(all, p2) = linsolver.solve(b(all, p2));
    }

    return x;
}

} // namespace ncorr

#endif
