#ifndef ARRAY2D_LINSOLVER_H
#define ARRAY2D_LINSOLVER_H

namespace ncorr {

namespace details {

template <typename T_container>
class base_linsolver {
    public:
        typedef typename T_container::value_type                             value_type;
        typedef typename T_container::reference                               reference;
        typedef typename T_container::size_type                               size_type;
        typedef typename T_container::difference_type                   difference_type;
        typedef typename T_container::coords                                     coords;
        typedef typename ::ncorr::details::container_traits<T_container>::nonconst_container    container;
        typedef typename ::ncorr::details::container_traits<T_container>::const_container const_container;

        friend container;

        base_linsolver() noexcept = default;
        base_linsolver(const base_linsolver&) = default;
        base_linsolver(base_linsolver&&) noexcept = default;
        base_linsolver& operator=(const base_linsolver&) = default;
        base_linsolver& operator=(base_linsolver&&) = default;
        virtual ~base_linsolver() noexcept = default;

        base_linsolver(const_container &A) : x_buf(A.width(), 1), A_factored_ptr(std::make_shared<container>(A)) { }

        virtual operator bool() = 0;

        container& backward_sub(const_container&, const_container&) const;
        container& forward_sub(const_container&, const_container&) const;
        virtual const_container& solve(const_container&) const = 0;

        virtual base_linsolver* clone() const = 0;

    protected:
        mutable container x_buf;
        std::shared_ptr<container> A_factored_ptr;
};

template <typename T_container>
class LU_linsolver final : public base_linsolver<T_container> {
    public:
        typedef typename base_linsolver<T_container>::value_type           value_type;
        typedef typename base_linsolver<T_container>::reference             reference;
        typedef typename base_linsolver<T_container>::size_type             size_type;
        typedef typename base_linsolver<T_container>::difference_type difference_type;
        typedef typename base_linsolver<T_container>::coords                   coords;
        typedef typename base_linsolver<T_container>::container             container;
        typedef typename base_linsolver<T_container>::const_container const_container;

        friend container;

        LU_linsolver() : full_rank() { }
        LU_linsolver(const LU_linsolver&) = default;
        LU_linsolver(LU_linsolver&&) noexcept = default;
        LU_linsolver& operator=(const LU_linsolver&) = default;
        LU_linsolver& operator=(LU_linsolver&&) = default;
        ~LU_linsolver() noexcept = default;

        LU_linsolver(const_container&);

        operator bool() override { return full_rank; }

        const_container& solve(const_container&) const override;

        LU_linsolver* clone() const override { return new LU_linsolver(*this); }

    private:
        std::shared_ptr<container> piv_ptr;
        bool full_rank;
};

template <typename T_container>
class QR_linsolver final : public base_linsolver<T_container> {
    public:
        typedef typename base_linsolver<T_container>::value_type           value_type;
        typedef typename base_linsolver<T_container>::reference             reference;
        typedef typename base_linsolver<T_container>::size_type             size_type;
        typedef typename base_linsolver<T_container>::difference_type difference_type;
        typedef typename base_linsolver<T_container>::coords                   coords;
        typedef typename base_linsolver<T_container>::container             container;
        typedef typename base_linsolver<T_container>::const_container const_container;

        friend container;

        QR_linsolver() : full_rank(), rank() { }
        QR_linsolver(const QR_linsolver&) = default;
        QR_linsolver(QR_linsolver&&) noexcept = default;
        QR_linsolver& operator=(const QR_linsolver&) = default;
        QR_linsolver& operator=(QR_linsolver&&) = default;
        ~QR_linsolver() noexcept = default;

        QR_linsolver(const_container&);

        operator bool() override { return full_rank; }

        std::pair<container, value_type> house(const_container&) const;
        const_container& solve(const_container&) const override;

        QR_linsolver* clone() const override { return new QR_linsolver(*this); }

    private:
        std::shared_ptr<container> piv_ptr;
        std::shared_ptr<container> beta_ptr;
        bool full_rank;
        difference_type rank;
};

template <typename T_container>
class CHOL_linsolver final : public base_linsolver<T_container> {
    public:
        typedef typename base_linsolver<T_container>::value_type           value_type;
        typedef typename base_linsolver<T_container>::reference             reference;
        typedef typename base_linsolver<T_container>::size_type             size_type;
        typedef typename base_linsolver<T_container>::difference_type difference_type;
        typedef typename base_linsolver<T_container>::coords                   coords;
        typedef typename base_linsolver<T_container>::container             container;
        typedef typename base_linsolver<T_container>::const_container const_container;

        friend container;

        CHOL_linsolver() : pos_def() { }
        CHOL_linsolver(const CHOL_linsolver&) = default;
        CHOL_linsolver(CHOL_linsolver&&) noexcept = default;
        CHOL_linsolver& operator=(const CHOL_linsolver&) = default;
        CHOL_linsolver& operator=(CHOL_linsolver&&) = default;
        ~CHOL_linsolver() noexcept = default;

        CHOL_linsolver(const_container &A);

        operator bool() override { return pos_def; }

        const_container& solve(const_container&) const override;

        CHOL_linsolver* clone() const override { return new CHOL_linsolver(*this); }

    private:
        bool pos_def;
};

template <typename T_linsolver>
class interface_linsolver final {
    public:
        typedef typename T_linsolver::value_type                 value_type;
        typedef typename T_linsolver::reference                   reference;
        typedef typename T_linsolver::size_type                   size_type;
        typedef typename T_linsolver::difference_type       difference_type;
        typedef typename T_linsolver::coords                         coords;
        typedef typename T_linsolver::container                   container;
        typedef typename T_linsolver::const_container       const_container;

        template <typename T_linsolver2>
        friend class interface_linsolver;
        friend container;

        interface_linsolver() noexcept : ptr(nullptr) { }
        interface_linsolver(const interface_linsolver &linsolver) : ptr(linsolver.ptr ? linsolver.ptr->clone() : nullptr) { }
        interface_linsolver(interface_linsolver &&linsolver) : ptr(std::move(linsolver.ptr)) { }
        interface_linsolver& operator=(const interface_linsolver &linsolver) { ptr.reset(linsolver.ptr ? linsolver.ptr->clone() : nullptr); return *this; }
        interface_linsolver& operator=(interface_linsolver &&linsolver) { ptr = std::move(linsolver.ptr); return *this; }
        ~interface_linsolver() noexcept = default;

        explicit interface_linsolver(T_linsolver *ptr) : ptr(ptr) { }

        operator bool() { return (*ptr); }

        const_container& solve(const_container &b) const { return ptr->solve(b); }

    private:
        std::shared_ptr<T_linsolver> ptr;
};

template <typename T_container>
typename base_linsolver<T_container>::container&
base_linsolver<T_container>::backward_sub(const_container &A, const_container &b) const {
    if (A.height() != A.width()) {
        throw std::invalid_argument("Attempted to perform backward substitution using an Array with size " + A.size_2D_string() +
                                    ". Array must be square.");
    }
    if (A.height() != b.height() || b.width() != 1) {
        throw std::invalid_argument("Attempted to perform backward substitution using an A with size " + A.size_2D_string() +
                                    " and a b with size " + b.size_2D_string() + ".");
    }

    std::copy(b.get_pointer(), b.get_pointer() + b.size(), x_buf.get_pointer());
    for (difference_type p = A.height() - 1; p > 0; --p) {
        if ((std::abs((*A_factored_ptr)(p,p)) < std::numeric_limits<value_type>::epsilon())) {
            x_buf(p) = 1;
        } else {
            x_buf(p) /= (*A_factored_ptr)(p,p);
        }

        for (difference_type p1 = p; p1 > 0; --p1) {
            x_buf(p1-1) -= x_buf(p) * (*A_factored_ptr)(p1-1,p);
        }
    }

    if ((std::abs((*A_factored_ptr)(0,0)) < std::numeric_limits<value_type>::epsilon())) {
        x_buf(0) = 1;
    } else {
        x_buf(0) /= (*A_factored_ptr)(0,0);
    }

    return x_buf;
}

template <typename T_container>
typename base_linsolver<T_container>::container&
base_linsolver<T_container>::forward_sub(const_container &A, const_container &b) const {
    if (A.height() != A.width()) {
        throw std::invalid_argument("Attempted to perform forward substitution using an Array with size " + A.size_2D_string() +
                                    ". Array must be square.");
    }
    if (A.height() != b.height() || b.width() != 1) {
        throw std::invalid_argument("Attempted to perform forward substitution using an A with size " + A.size_2D_string() +
                                    " and a b with size " + b.size_2D_string() + ".");
    }

    std::copy(b.get_pointer(), b.get_pointer() + b.size(), x_buf.get_pointer());
    for (difference_type p = 0; p < A.height() - 1; ++p) {
        if ((std::abs((*A_factored_ptr)(p,p)) < std::numeric_limits<value_type>::epsilon())) {
            x_buf(p) = 1;
        } else {
            x_buf(p) /= (*A_factored_ptr)(p,p);
        }

        for (difference_type p1 = p + 1; p1 < A_factored_ptr->height(); ++p1) {
            x_buf(p1) -= x_buf(p) * (*A_factored_ptr)(p1,p);
        }
    }

    if ((std::abs((*A_factored_ptr)(last,last)) < std::numeric_limits<value_type>::epsilon())) {
        x_buf(last) = 1;
    } else {
        x_buf(last) /= (*A_factored_ptr)(last,last);
    }

    return x_buf;
}

template <typename T_container>
LU_linsolver<T_container>::LU_linsolver(const_container &A)
    : base_linsolver<container>(A), piv_ptr(std::make_shared<container>(A.height(), 1)), full_rank(true) {
    if (A.height() != A.width()) {
        throw std::invalid_argument("Attempted to perform LU decomposition on array of size " + A.size_2D_string() +
                                    ". Array must be square.");
    }

    for (difference_type p = 0; p < this->A_factored_ptr->height() - 1; ++p) {
        difference_type mu = std::max_element(&(*this->A_factored_ptr)(p,p), &(*this->A_factored_ptr)(0,p) + this->A_factored_ptr->height()) - &(*this->A_factored_ptr)(0,p);
        (*this->piv_ptr)(p) = mu;

        for (difference_type p2 = p; p2 < this->A_factored_ptr->width(); ++p2) {
            value_type buf = (*this->A_factored_ptr)(p,p2);
            (*this->A_factored_ptr)(p,p2) = (*this->A_factored_ptr)(mu,p2);
            (*this->A_factored_ptr)(mu,p2) = buf;
        }

        if (std::abs((*this->A_factored_ptr)(p,p)) < std::numeric_limits<value_type>::epsilon()) {
            full_rank = false;
        } else {
            for (difference_type p1 = p + 1; p1 < this->A_factored_ptr->height(); ++p1) {
                (*this->A_factored_ptr)(p1,p) /= (*this->A_factored_ptr)(p,p);
            }

            for (difference_type p2 = p + 1; p2 < this->A_factored_ptr->width(); ++p2) {
                for (difference_type p1 = p + 1; p1 < this->A_factored_ptr->height(); ++p1) {
                    (*this->A_factored_ptr)(p1,p2) -= (*this->A_factored_ptr)(p1,p) * (*this->A_factored_ptr)(p,p2);
                }
            }
        }
    }

    if (std::abs((*this->A_factored_ptr)(last,last)) < std::numeric_limits<value_type>::epsilon()) {
        full_rank = false;
    }
}

template <typename T_container>
typename LU_linsolver<T_container>::const_container&
LU_linsolver<T_container>::solve(const_container &b) const {
    if ((*this->A_factored_ptr).height() != b.height() || b.width() != 1) {
        throw std::invalid_argument("Attempted to solve LU decomposition using b of size " + b.size_2D_string() +
                                    " on an Array of size " + (*this->A_factored_ptr).size_2D_string() + ".");
    }

    std::copy(b.get_pointer(), b.get_pointer() + b.size(), this->x_buf.get_pointer());
    for (difference_type p = 0; p < b.height() - 1; ++p) {
        value_type buf = this->x_buf((*piv_ptr)(p));
        this->x_buf((*piv_ptr)(p)) = this->x_buf(p);
        this->x_buf(p) = buf;

        for (difference_type p1 = p + 1; p1 < b.height(); ++p1) {
            this->x_buf(p1) -= this->x_buf(p) * (*this->A_factored_ptr)(p1,p);
        }
    }

    return this->backward_sub((*this->A_factored_ptr), this->x_buf);
}

template <typename T_container>
QR_linsolver<T_container>::QR_linsolver(const_container &A)
    : base_linsolver<container>(A),
      piv_ptr(std::make_shared<container>(A.width(), 1)),
      beta_ptr(std::make_shared<container>(A.width(), 1)),
      full_rank(true),
      rank(0) {
    container c(A.width(), 1);
    for (difference_type p2 = 0; p2 < A.width(); ++p2) {
        c(p2) = dot(A(all,p2), A(all,p2));
    }
    value_type tau = max(c);
    difference_type k = find(c == tau);
    difference_type min_dim = std::min(A.height(), A.width());
    while (std::abs(tau) >= std::numeric_limits<value_type>::epsilon() && tau > 0) {
        ++rank;
        (*piv_ptr)(rank - 1) = k;

        container col_buf = (*this->A_factored_ptr)(all, k);
        (*this->A_factored_ptr)(all, k) = (*this->A_factored_ptr)(all, rank - 1);
        (*this->A_factored_ptr)(all, rank - 1) = col_buf;

        value_type buf_c = c(k);
        c(k) = c(rank - 1);
        c(rank - 1) = buf_c;

        auto h_pair = house((*this->A_factored_ptr)({rank - 1, last}, rank - 1));
        (*beta_ptr)(rank - 1) = h_pair.second;

        (*this->A_factored_ptr)({rank - 1, last}, {rank - 1, last}) =
            (*this->A_factored_ptr)({rank - 1, last}, {rank - 1, last}) -
            h_pair.second * h_pair.first * (t(h_pair.first) * (*this->A_factored_ptr)({rank - 1, last}, {rank - 1, last}));
        (*this->A_factored_ptr)({rank, last}, rank - 1) = h_pair.first({1, (*this->A_factored_ptr).height() - rank});

        for (difference_type p2 = rank; p2 < (*this->A_factored_ptr).width(); ++p2) {
            c(p2) = c(p2) - std::pow((*this->A_factored_ptr)(rank - 1, p2), 2);
        }

        if (rank < min_dim) {
            tau = max(c({rank, last}));
            k = find(c == tau, rank);
        } else {
            tau = 0;
        }
    }
    full_rank = (rank == (*this->A_factored_ptr).width());
}

template <typename T_container>
std::pair<typename QR_linsolver<T_container>::container, typename QR_linsolver<T_container>::value_type>
QR_linsolver<T_container>::house(const_container &x) const {
    if (x.width() != 1) {
        throw std::invalid_argument("Attempted to get household vector for vector of size " + x.size_2D_string() +
                                    ". Vector must be a column vector.");
    }

    value_type sigma = dot(x({1,last}), x({1,last}));
    container v(x.height(), 1, 1);
    v({1,last}) = x({1,last});
    value_type beta = 0;
    if (std::abs(sigma) > std::numeric_limits<value_type>::epsilon()) {
        value_type mu = std::sqrt(std::pow(x(0), 2) + sigma);
        if (std::abs(x(0)) < std::numeric_limits<value_type>::epsilon() || x(0) < 0) {
            v(0) = x(0) - mu;
        } else {
            v(0) = -sigma / (x(0) + mu);
        }
        beta = 2 * std::pow(v(0), 2) / (sigma + std::pow(v(0), 2));
        v = v / v(0);
    }

    return {std::move(v), std::move(beta)};
}

template <typename T_container>
typename QR_linsolver<T_container>::const_container&
QR_linsolver<T_container>::solve(const_container &b) const {
    if ((*this->A_factored_ptr).height() != b.height() || b.width() != 1) {
        throw std::invalid_argument("Attempted to solve QR decomposition using b of size " + b.size_2D_string() +
                                    " on an Array of size " + (*this->A_factored_ptr).size_2D_string() + ".");
    }

    container y(b);
    container v(b.height(), 1);
    for (difference_type p = 0; p < rank; ++p) {
        v(p) = 1;
        v({p + 1, last}) = (*this->A_factored_ptr)({p + 1, last}, p);
        y({p, last}) = y({p, last}) - (*beta_ptr)(p) * v({p, last}) * (t(v({p, last})) * y({p, last}));
    }

    this->backward_sub((*this->A_factored_ptr)({0, rank - 1}, {0, rank - 1}), y({0, rank - 1}));

    this->x_buf({rank, last}) = 0;
    for (difference_type p = rank; p > 0; --p) {
        value_type buf = this->x_buf((*piv_ptr)(p - 1));
        this->x_buf((*piv_ptr)(p - 1)) = this->x_buf(p - 1);
        this->x_buf(p - 1) = buf;
    }

    return this->x_buf;
}

template <typename T_container>
CHOL_linsolver<T_container>::CHOL_linsolver(const_container &A)
    : base_linsolver<container>(A), pos_def(true) {
    if (A.height() != A.width()) {
        throw std::invalid_argument("Attempted to perform Cholesky decomposition on array of size " + A.size_2D_string() +
                                    ". Array must be square.");
    }

    for (difference_type p = 0; p < (*this->A_factored_ptr).height(); ++p) {
        if (p > 0) {
            for (difference_type p1 = this->A_factored_ptr->height() - 1; p1 >= p; --p1) {
                double buf = 0.0;
                for (difference_type p2 = 0; p2 < p; ++p2) {
                    buf += (*this->A_factored_ptr)(p1,p2) * (*this->A_factored_ptr)(p,p2);
                }
                (*this->A_factored_ptr)(p1,p) -= buf;
            }
        }

        if ((*this->A_factored_ptr)(p,p) >= std::numeric_limits<value_type>::epsilon()) {
            double diag_sqrt = std::sqrt((*this->A_factored_ptr)(p,p));
            for (difference_type p1 = p; p1 < this->A_factored_ptr->height(); ++p1) {
                (*this->A_factored_ptr)(p1,p) /= diag_sqrt;
            }
        } else {
            pos_def = false;
            return;
        }
    }

    for (difference_type p2 = 0; p2 < this->A_factored_ptr->width() - 1; ++p2) {
        for (difference_type p1 = p2 + 1; p1 < this->A_factored_ptr->height(); ++p1) {
            (*this->A_factored_ptr)(p2,p1) = (*this->A_factored_ptr)(p1,p2);
        }
    }
}

template <typename T_container>
typename CHOL_linsolver<T_container>::const_container&
CHOL_linsolver<T_container>::solve(const_container &b) const {
    if (!pos_def) {
        throw std::invalid_argument("Attempted to solve Cholesky decomposition with a non positive definite matrix.");
    }
    if ((*this->A_factored_ptr).height() != b.height() || b.width() != 1) {
        throw std::invalid_argument("Attempted to solve Cholesky decomposition using b of size " + b.size_2D_string() +
                                    " on an Array of size " + (*this->A_factored_ptr).size_2D_string() + ".");
    }

    return this->backward_sub((*this->A_factored_ptr), this->forward_sub((*this->A_factored_ptr), b));
}

} // namespace details

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
    auto linsolve = ((this->h == this->w) ? get_linsolver(LINSOLVER::LU) : get_linsolver(LINSOLVER::QR));

    Array2D x(this->w, b.width());
    for (difference_type p2 = 0; p2 < b.width(); ++p2) {
        x(all, p2) = linsolve.solve(b(all, p2));
    }

    return x;
}

} // namespace ncorr

#endif
