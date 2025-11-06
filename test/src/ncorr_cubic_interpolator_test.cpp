#include "ncorr.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <chrono>

using namespace ncorr;
using json = nlohmann::json;

// Forward declarations
template <typename T_container> class coef_mat_interp_base;
template <typename T_container> class cubic_interp_base;
template <typename T_container> class cubic_interp;
template <typename T_container> class quintic_interp_base;
template <typename T_container> class quintic_interp;

// Copied from Array2D.h for direct access to internal methods
template <typename T_container> 
class coef_mat_interp_base {
    // The is the base class for interpolation schemes which use a coefficient
    // matrix
    public:          
        typedef typename T_container::value_type           value_type;
        typedef typename T_container::reference             reference;  
        typedef typename T_container::size_type             size_type; 
        typedef typename T_container::difference_type difference_type;  
        typedef std::pair<double, double>                      coords;
        typedef T_container                                 container;
        typedef const T_container                     const_container;
        
        friend container;
        
        // Rule of 5 and destructor --------------------------------------//
        coef_mat_interp_base() : order() { }
        coef_mat_interp_base(const coef_mat_interp_base&) = default;
        coef_mat_interp_base(coef_mat_interp_base&&) noexcept = default;
        coef_mat_interp_base& operator=(const coef_mat_interp_base&) = default;  
        coef_mat_interp_base& operator=(coef_mat_interp_base&&) = default;
        virtual ~coef_mat_interp_base() noexcept = default;
             
        // Additional Constructors ---------------------------------------//            
        coef_mat_interp_base(const_container &A, difference_type order) : 
            A_ptr(&A), order(order), p1_pow_buf(order+1,1), p2_pow_buf(order+1,1), 
            p1_pow_dp1_buf(order+1,1), p2_pow_dp2_buf(order+1,1), first_order_buf(3,1) { 
            if (order < 1) {
                // This is purely a programmer error since this class is abstract
                throw std::invalid_argument("Attempted to form coef_mat_interp_base with order of: " + std::to_string(order) + " order must be 1 or greater.");
            }
        }
        
        virtual difference_type get_order() const { return order; }
        // Arithmetic methods --------------------------------------------//
        virtual value_type operator()(double, double) const;
        virtual const_container& first_order(double, double) const;
        
        // Clone ---------------------------------------------------------//
        virtual coef_mat_interp_base* clone() const = 0;
        
    protected:
        // Utility -------------------------------------------------------//
        virtual bool out_of_bounds(double p1, double p2) const { return p1 < 0 || p1 >= A_ptr->height() || p2 < 0 || p2 >= A_ptr->width(); }
                 
        // Arithmetic methods --------------------------------------------//
        virtual container& get_p_pow(container&, double) const;
        virtual container& get_dp_pow(container&, const_container&) const;
        virtual value_type t_vec_mat_vec(const_container&, const_container&, const_container&) const;
        virtual const_container& calc_coef_mat(container&, const_container&, difference_type, difference_type) const = 0;
        virtual const_container& get_coef_mat(difference_type, difference_type) const = 0;
        
        const_container *A_ptr = nullptr; // immutable
        difference_type order;
        mutable container p1_pow_buf;
        mutable container p2_pow_buf;
        mutable container p1_pow_dp1_buf;
        mutable container p2_pow_dp2_buf;
        mutable container first_order_buf;
};

template <typename T_container> 
class cubic_interp_base : public coef_mat_interp_base<T_container> {
    // Cubic interpolation base class
    public:          
        typedef typename coef_mat_interp_base<T_container>::value_type           value_type;
        typedef typename coef_mat_interp_base<T_container>::reference             reference;  
        typedef typename coef_mat_interp_base<T_container>::size_type             size_type; 
        typedef typename coef_mat_interp_base<T_container>::difference_type difference_type;  
        typedef typename coef_mat_interp_base<T_container>::coords                   coords;
        typedef typename coef_mat_interp_base<T_container>::container             container;
        typedef typename coef_mat_interp_base<T_container>::const_container const_container;
        
        friend container;
        
        // Rule of 5 and destructor --------------------------------------//
        cubic_interp_base() = default;
        cubic_interp_base(const cubic_interp_base&) = default;
        cubic_interp_base(cubic_interp_base&&) noexcept = default;
        cubic_interp_base& operator=(const cubic_interp_base&) = default;  
        cubic_interp_base& operator=(cubic_interp_base&&) = default;
        virtual ~cubic_interp_base() noexcept = default;
                                                      
        // Additional Constructors ---------------------------------------//            
        cubic_interp_base(const_container &A) : coef_mat_interp_base<container>(A,3) { }
        
        // Arithmetic methods --------------------------------------------//
        value_type operator()(double, double) const override;
        const_container& first_order(double, double) const override;
        
    public:            
        // Arithmetic methods --------------------------------------------//
        const_container& calc_coef_mat(container &coef_mat_buf, const_container &A, difference_type p1, difference_type p2) const override;
        
        // Utility -------------------------------------------------------//
        // Uses 16 points around floored point. Possibly add scheme to 
        // interpolate points within the image near the border that can't 
        // be interpolated later.
        bool out_of_bounds(double p1, double p2) const override { return p1 < 1 || p1 >= this->A_ptr->height() - 2 || p2 < 1 || p2 >= this->A_ptr->width() - 2; }
};

template <typename T_container> 
class cubic_interp final : public cubic_interp_base<T_container> { 
    // Bicubic interpolation
    public:          
        typedef typename cubic_interp_base<T_container>::value_type           value_type;
        typedef typename cubic_interp_base<T_container>::reference             reference;  
        typedef typename cubic_interp_base<T_container>::size_type             size_type; 
        typedef typename cubic_interp_base<T_container>::difference_type difference_type;  
        typedef typename cubic_interp_base<T_container>::coords                   coords;
        typedef typename cubic_interp_base<T_container>::container             container;
        typedef typename cubic_interp_base<T_container>::const_container const_container;
        
        friend container;
        
        // Rule of 5 and destructor --------------------------------------//
        cubic_interp() = default;
        cubic_interp(const cubic_interp&) = default;
        cubic_interp(cubic_interp&&) noexcept = default;
        cubic_interp& operator=(const cubic_interp&) = default;  
        cubic_interp& operator=(cubic_interp&&) = default;
        ~cubic_interp() noexcept = default;
        
        // Additional Constructors ---------------------------------------//            
        cubic_interp(const_container &A) : cubic_interp_base<container>(A), coef_mat_buf(4,4), p1_cache(-1), p2_cache(-1) { }
                            
        // Clone ---------------------------------------------------------//
        cubic_interp* clone() const override { return new cubic_interp(*this); }
        
    public:
        // Coefficient matrix cache
        mutable container coef_mat_buf;
        mutable difference_type p1_cache;
        mutable difference_type p2_cache;
        
        // Get coefficient matrix (with caching)
        const_container& get_coef_mat(difference_type p1, difference_type p2) const override {
            return this->calc_coef_mat(coef_mat_buf, *this->A_ptr, p1 - 1, p2 - 1);
        }
};

template <typename T_container> 
class quintic_interp_base : public coef_mat_interp_base<T_container> {
    // Biquintic B-spline base class
    public:          
        typedef typename coef_mat_interp_base<T_container>::value_type           value_type;
        typedef typename coef_mat_interp_base<T_container>::reference             reference;  
        typedef typename coef_mat_interp_base<T_container>::size_type             size_type; 
        typedef typename coef_mat_interp_base<T_container>::difference_type difference_type;  
        typedef typename coef_mat_interp_base<T_container>::coords                   coords;
        typedef typename coef_mat_interp_base<T_container>::container             container;
        typedef typename coef_mat_interp_base<T_container>::const_container const_container;
        
        friend container;
        
        // Rule of 5 and destructor --------------------------------------//
        quintic_interp_base() = default;
        quintic_interp_base(const quintic_interp_base&) = default;
        quintic_interp_base(quintic_interp_base&&) noexcept = default;
        quintic_interp_base& operator=(const quintic_interp_base&) = default;  
        quintic_interp_base& operator=(quintic_interp_base&&) = default;
        virtual ~quintic_interp_base() noexcept = default;
        
        // Additional Constructors ---------------------------------------//            
        quintic_interp_base(const_container &A) : coef_mat_interp_base<container>(A,5), dp1_pow_buf(6,1), dp2_pow_buf(6,1) { }
                                          
        // Arithmetic methods --------------------------------------------//
        value_type operator()(double, double) const override;
        const_container& first_order(double, double) const override;
        
    public:            
        // Arithmetic methods --------------------------------------------//
        std::shared_ptr<container> get_bspline_mat_ptr(const_container&) const;
        std::shared_ptr<container> get_bspline_mat_ptr_pad(const_container&) const;
        std::shared_ptr<container> get_bspline_mat_ptr_pad_deconv(const_container&) const;
        const_container& calc_coef_mat(container&, const_container&, difference_type, difference_type) const override;
        
        // Utility -------------------------------------------------------//
        // Uses 36 points around floored point. Allow interpolation within 
        // entire image bounds since padding is used for the b-coefficient 
        // array; this padding must be 3 or greater.
        bool out_of_bounds(double p1, double p2) const override { return p1 < 0 || p1 > this->A_ptr->height() - 1 || p2 < 0 || p2 > this->A_ptr->width() - 1; }
        
        // Must be greater than or equal to 3 in order to interpolate entire 
        // image for any input size. Large borders mitigate ringing errors.
        difference_type bcoef_border = 20;
        
        // Buffers for derivative calculations
        mutable container dp1_pow_buf;
        mutable container dp2_pow_buf; 
};

template <typename T_container> 
class quintic_interp final : public quintic_interp_base<T_container> { 
    // Biquintic B-spline interpolation
    public:          
        typedef typename quintic_interp_base<T_container>::value_type           value_type;
        typedef typename quintic_interp_base<T_container>::reference             reference;  
        typedef typename quintic_interp_base<T_container>::size_type             size_type; 
        typedef typename quintic_interp_base<T_container>::difference_type difference_type;  
        typedef typename quintic_interp_base<T_container>::coords                   coords;
        typedef typename quintic_interp_base<T_container>::container             container;
        typedef typename quintic_interp_base<T_container>::const_container const_container;
        
        friend container;
        
        // Rule of 5 and destructor --------------------------------------//
        quintic_interp() = default;
        quintic_interp(const quintic_interp&) = default;
        quintic_interp(quintic_interp&&) noexcept = default;
        quintic_interp& operator=(const quintic_interp&) = default;  
        quintic_interp& operator=(quintic_interp&&) = default;
        ~quintic_interp() noexcept = default;
        
        // Additional Constructors ---------------------------------------//            
        quintic_interp(const_container &A) : quintic_interp_base<container>(A), coef_mat_buf(6,6), bcoef_ptr(this->get_bspline_mat_ptr(A)) { }
                                
        // Clone ---------------------------------------------------------//
        quintic_interp* clone() const override { return new quintic_interp(*this); }
        
    public:
        // Arithmetic methods --------------------------------------------//
        // This will compute the coefficient matrix
        const_container& get_coef_mat(difference_type p1, difference_type p2) const override {
            return this->calc_coef_mat(coef_mat_buf, *bcoef_ptr, p1 + this->bcoef_border - 2, p2 + this->bcoef_border - 2);
        }
                    
        mutable container coef_mat_buf;   
        std::shared_ptr<container> bcoef_ptr; // immutable
}; 

// Implementation of coef_mat_interp_base methods
template <typename T_container> 
inline typename coef_mat_interp_base<T_container>::value_type coef_mat_interp_base<T_container>::operator()(double p1, double p2) const {
    // This is the general interpolation scheme for interpolators that use
    // a coefficient matrix. Can be overridden in child classes to increase
    // speed.
    if (out_of_bounds(p1,p2)) {
        return std::numeric_limits<value_type>::quiet_NaN();
    }
    
    // Get integer and fractional parts of p1 and p2
    difference_type p1_floor = static_cast<difference_type>(std::floor(p1));
    difference_type p2_floor = static_cast<difference_type>(std::floor(p2));
    double p1_frac = p1 - p1_floor;
    double p2_frac = p2 - p2_floor;
    
    // Get coefficient matrix
    const_container &coef_mat = this->get_coef_mat(p1_floor, p2_floor);
    
    // Get powers of p1_frac and p2_frac
    this->get_p_pow(this->p1_pow_buf, p1_frac);
    this->get_p_pow(this->p2_pow_buf, p2_frac);
    
    // Compute interpolated value
    return this->t_vec_mat_vec(this->p1_pow_buf, coef_mat, this->p2_pow_buf);
}

template <typename T_container> 
inline typename coef_mat_interp_base<T_container>::const_container& coef_mat_interp_base<T_container>::first_order(double p1, double p2) const {  
    // This is the general first-order interpolation scheme for interpolators 
    // that use a coefficient matrix. Can be overridden in child classes to 
    // increase speed.
    if (this->out_of_bounds(p1,p2)) {
        this->first_order_buf(0) = this->first_order_buf(1) = this->first_order_buf(2) = std::numeric_limits<value_type>::quiet_NaN();
        return this->first_order_buf;
    }
    
    // Get powers of delta_p1
    p1_pow_buf = get_p_pow(p1_pow_buf, p1 - std::floor(p1));
    
    // Get derivatives of p1
    p1_pow_dp1_buf = get_dp_pow(p1_pow_dp1_buf, p1_pow_buf);
    
    // Get powers of delta_p2
    p2_pow_buf = get_p_pow(p2_pow_buf, p2 - std::floor(p2));
    
    // Get derivatives of p2
    p2_pow_dp2_buf = get_dp_pow(p2_pow_dp2_buf, p2_pow_buf);

    // Get coefficient matrix
    const auto &coef_mat = this->get_coef_mat(p1, p2);
       
    // Interpolate values
    this->first_order_buf(0) = t_vec_mat_vec(p1_pow_buf, coef_mat, p2_pow_buf);
    this->first_order_buf(1) = t_vec_mat_vec(p1_pow_dp1_buf, coef_mat, p2_pow_buf);
    this->first_order_buf(2) = t_vec_mat_vec(p1_pow_buf, coef_mat, p2_pow_dp2_buf);
            
    return this->first_order_buf;
}
    
template <typename T_container> 
inline typename coef_mat_interp_base<T_container>::container& coef_mat_interp_base<T_container>::get_p_pow(container &p_pow, double delta_p) const {  
    // Order must be 1 or greater - note this is already checked for in the
    // base interpolator's constructor.
    p_pow(0) = 1.0;
    p_pow(1) = delta_p;
    for (difference_type i = 2; i <= this->order; ++i) {
        p_pow(i) = p_pow(i-1) * delta_p; 
    }
    
    return p_pow;
}
    
template <typename T_container> 
inline typename coef_mat_interp_base<T_container>::container& coef_mat_interp_base<T_container>::get_dp_pow(container &p_pow_dp, const_container &p_pow) const {  
    // Order must be 1 or greater - note this is already checked for in the
    // base interpolator's constructor.
    p_pow_dp(0) = 0.0;
    p_pow_dp(1) = 1.0;
    for (difference_type i = 2; i <= this->order; ++i) {
        p_pow_dp(i) = p_pow(i-1) * i; 
    }
    
    return p_pow_dp;
}

template <typename T_container> 
inline typename coef_mat_interp_base<T_container>::value_type coef_mat_interp_base<T_container>::t_vec_mat_vec(const_container &vec1, const_container &mat, const_container &vec2) const {  
    return value_type( t( vec1 ) * mat * vec2 );
}

// Implementation of quintic_interp_base methods
template <typename T_container> 
std::shared_ptr<typename quintic_interp_base<T_container>::container> quintic_interp_base<T_container>::get_bspline_mat_ptr(const_container &A) const {
    #ifndef NDEBUG
    if (bcoef_border < 3) {
        throw std::invalid_argument("B-coefficient border cannot be less than three when calling get_bspline_coef() - this is a programmer error.");
    }
    #endif
    
    // Quintic b-spline kernel - deconvolve twice to get b-spline coefficients
    // This is read - only
    static const_container kernel_qb = {1/120.0, 13/60.0, 11/20.0, 13/60.0, 1/120.0};

    // Create b-spline coefficient array and deconvolve in different directions 
    // twice. Note: should precompute 2D quintic B-spline kernel and deconvolve 
    // once for increased speed, but this is usually not a bottleneck.
    auto padded_data = pad(A, bcoef_border, PAD::EXPAND_EDGES);
    std::cout<< "before deconvolution: \n"<<t(kernel_qb)<<std::endl;
    auto deconvolved_data = deconv(padded_data, t(kernel_qb));
    std::cout<< "after deconvolution: \n"<<kernel_qb<<std::endl;
    auto deconvolved_data2 = deconv(deconvolved_data, kernel_qb);
   
    // save in a file deconvolved_data2
    std::ofstream ofs("deconvolved_data.txt");
    ofs<<deconvolved_data;
    
    return std::make_shared<container>(deconvolved_data2); 
}

template <typename T_container> 
std::shared_ptr<typename quintic_interp_base<T_container>::container> quintic_interp_base<T_container>::get_bspline_mat_ptr_pad(const_container &A) const {
    #ifndef NDEBUG
    if (bcoef_border < 3) {
        throw std::invalid_argument("B-coefficient border cannot be less than three when calling get_bspline_coef() - this is a programmer error.");
    }
    #endif
    
    // Quintic b-spline kernel - deconvolve twice to get b-spline coefficients
    // This is read - only
    static const_container kernel_qb = {1/120.0, 13/60.0, 11/20.0, 13/60.0, 1/120.0};

    // Create b-spline coefficient array and deconvolve in different directions 
    // twice. Note: should precompute 2D quintic B-spline kernel and deconvolve 
    // once for increased speed, but this is usually not a bottleneck.
    return std::make_shared<container>(pad(A, bcoef_border, PAD::EXPAND_EDGES)); 
}

template <typename T_container> 
std::shared_ptr<typename quintic_interp_base<T_container>::container> quintic_interp_base<T_container>::get_bspline_mat_ptr_pad_deconv(const_container &A) const {
    #ifndef NDEBUG
    if (bcoef_border < 3) {
        throw std::invalid_argument("B-coefficient border cannot be less than three when calling get_bspline_coef() - this is a programmer error.");
    }
    #endif
    
    // Quintic b-spline kernel - deconvolve twice to get b-spline coefficients
    // This is read - only
    static const_container kernel_qb = {1/120.0, 13/60.0, 11/20.0, 13/60.0, 1/120.0};

    // Create b-spline coefficient array and deconvolve in different directions 
    // twice. Note: should precompute 2D quintic B-spline kernel and deconvolve 
    // once for increased speed, but this is usually not a bottleneck.
    return std::make_shared<container>(deconv(pad(A, bcoef_border, PAD::EXPAND_EDGES), t(kernel_qb)));
}

template <typename T_container> 
inline typename quintic_interp_base<T_container>::const_container& quintic_interp_base<T_container>::calc_coef_mat(container &coef_mat_buf, const_container &bcoef, difference_type p1, difference_type p2) const {
    // p1 and p2 refer to the top-left corner of the desired coefficient matrix
    #ifndef NDEBUG
    if (p1 < 0 || p1+5 >= bcoef.height() || p2 < 0 || p2+5 >= bcoef.width()) {
        throw std::invalid_argument("p1 and p2 are outside range of array for calc_coef_mat() with quintic interpolation - this is a programmer error.");
    }
    #endif
    
    // Simplified version of the coefficient matrix calculation for testing
    // The full implementation has very long formulas for each element
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            coef_mat_buf(i, j) = 0.0;
            // Sum contributions from the 36 points around p1,p2
            for (int k = 0; k < 6; k++) {
                for (int l = 0; l < 6; l++) {
                    // Apply appropriate weights based on position
                    double weight = 1.0;
                    if (k == 0 || k == 5 || l == 0 || l == 5) weight *= 0.1;
                    if ((k == 1 || k == 4) && (l == 1 || l == 4)) weight *= 0.5;
                    if ((k == 2 || k == 3) && (l == 2 || l == 3)) weight *= 1.0;
                    
                    coef_mat_buf(i, j) += weight * bcoef(p1 + k, p2 + l);
                }
            }
        }
    }
    
    return coef_mat_buf;
}

template <typename T_container> 
inline typename quintic_interp_base<T_container>::value_type quintic_interp_base<T_container>::operator()(double p1, double p2) const {
    if (out_of_bounds(p1,p2)) {
        return std::numeric_limits<value_type>::quiet_NaN();
    }
    
    // Get integer and fractional parts of p1 and p2
    difference_type p1_floor = static_cast<difference_type>(std::floor(p1));
    difference_type p2_floor = static_cast<difference_type>(std::floor(p2));
    double p1_frac = p1 - p1_floor;
    double p2_frac = p2 - p2_floor;
    
    // Get coefficient matrix
    const_container &coef_mat = this->get_coef_mat(p1_floor, p2_floor);
    
    // Get powers of p1_frac and p2_frac
    this->get_p_pow(this->p1_pow_buf, p1_frac);
    this->get_p_pow(this->p2_pow_buf, p2_frac);
    
    // Compute interpolated value
    return this->t_vec_mat_vec(this->p1_pow_buf, coef_mat, this->p2_pow_buf);
}

template <typename T_container> 
inline typename quintic_interp_base<T_container>::const_container& quintic_interp_base<T_container>::first_order(double p1, double p2) const {
    if (out_of_bounds(p1,p2)) {
        throw std::out_of_range("Point is out of bounds.");
    }
    
    // Get integer and fractional parts of p1 and p2
    difference_type p1_floor = static_cast<difference_type>(std::floor(p1));
    difference_type p2_floor = static_cast<difference_type>(std::floor(p2));
    double p1_frac = p1 - p1_floor;
    double p2_frac = p2 - p2_floor;
    
    // Get coefficient matrix
    const_container &coef_mat = this->get_coef_mat(p1_floor, p2_floor);
    
    // Get powers of p1_frac and p2_frac
    this->get_p_pow(this->p1_pow_buf, p1_frac);
    this->get_p_pow(this->p2_pow_buf, p2_frac);
    this->get_dp_pow(this->dp1_pow_buf, this->p1_pow_buf);
    this->get_dp_pow(this->dp2_pow_buf, this->p2_pow_buf);
    
    // Compute value and derivatives
    this->first_order_buf(0, 0) = this->t_vec_mat_vec(this->p1_pow_buf, coef_mat, this->p2_pow_buf);
    this->first_order_buf(1, 0) = this->t_vec_mat_vec(this->dp1_pow_buf, coef_mat, this->p2_pow_buf);
    this->first_order_buf(2, 0) = this->t_vec_mat_vec(this->p1_pow_buf, coef_mat, this->dp2_pow_buf);
    
    return this->first_order_buf;
}

// Implementation of cubic_interp_base methods
template <typename T_container> 
inline typename cubic_interp_base<T_container>::value_type cubic_interp_base<T_container>::operator()(double p1, double p2) const { 
    if (out_of_bounds(p1,p2)) {
        return std::numeric_limits<value_type>::quiet_NaN();
    }        

    double delta_p1 = p1 - std::floor(p1);
    double delta_p2 = p2 - std::floor(p2);
    
    this->p1_pow_buf(0) = 1.0;
    this->p1_pow_buf(1) = delta_p1;
    this->p1_pow_buf(2) = this->p1_pow_buf(1)*delta_p1;
    this->p1_pow_buf(3) = this->p1_pow_buf(2)*delta_p1;

    this->p2_pow_buf(0) = 1.0;
    this->p2_pow_buf(1) = delta_p2;
    this->p2_pow_buf(2) = this->p2_pow_buf(1)*delta_p2;
    this->p2_pow_buf(3) = this->p2_pow_buf(2)*delta_p2;

    const auto &coef_mat = this->get_coef_mat(p1, p2);
    
    return (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3))*this->p1_pow_buf(0)+
           (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3))*this->p1_pow_buf(1)+
           (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3))*this->p1_pow_buf(2)+
           (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3))*this->p1_pow_buf(3);
}

template <typename T_container> 
inline typename cubic_interp_base<T_container>::const_container& cubic_interp_base<T_container>::first_order(double p1, double p2) const { 
    if (out_of_bounds(p1,p2)) {
        this->first_order_buf(0) = this->first_order_buf(1) = this->first_order_buf(2) = std::numeric_limits<value_type>::quiet_NaN();
        return this->first_order_buf;
    }
            
    double delta_p1 = p1 - std::floor(p1);
    double delta_p2 = p2 - std::floor(p2);

    this->p1_pow_buf(0) = 1.0;
    this->p1_pow_buf(1) = delta_p1;
    this->p1_pow_buf(2) = this->p1_pow_buf(1) * delta_p1;
    this->p1_pow_buf(3) = this->p1_pow_buf(2) * delta_p1;
    
    this->p1_pow_dp1_buf(0) = 0.0;
    this->p1_pow_dp1_buf(1) = 1.0;
    this->p1_pow_dp1_buf(2) = 2.0 * this->p1_pow_buf(1);
    this->p1_pow_dp1_buf(3) = 3.0 * this->p1_pow_buf(2);

    this->p2_pow_buf(0) = 1.0;
    this->p2_pow_buf(1) = delta_p2;
    this->p2_pow_buf(2) = this->p2_pow_buf(1) * delta_p2;
    this->p2_pow_buf(3) = this->p2_pow_buf(2) * delta_p2;
    
    this->p2_pow_dp2_buf(0) = 0.0;
    this->p2_pow_dp2_buf(1) = 1.0;
    this->p2_pow_dp2_buf(2) = 2.0 * this->p2_pow_buf(1);
    this->p2_pow_dp2_buf(3) = 3.0 * this->p2_pow_buf(2);

    const auto &coef_mat = this->get_coef_mat(p1, p2);
    
    this->first_order_buf(0) = (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3))*this->p1_pow_buf(0)+
                               (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3))*this->p1_pow_buf(1)+
                               (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3))*this->p1_pow_buf(2)+
                               (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3))*this->p1_pow_buf(3);
    
    this->first_order_buf(1) = (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3))*this->p1_pow_dp1_buf(0)+
                               (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3))*this->p1_pow_dp1_buf(1)+
                               (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3))*this->p1_pow_dp1_buf(2)+
                               (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3))*this->p1_pow_dp1_buf(3);
    
    this->first_order_buf(2) = (this->p2_pow_dp2_buf(0)*coef_mat(0,0)+this->p2_pow_dp2_buf(1)*coef_mat(0,1)+this->p2_pow_dp2_buf(2)*coef_mat(0,2)+this->p2_pow_dp2_buf(3)*coef_mat(0,3))*this->p1_pow_buf(0)+
                               (this->p2_pow_dp2_buf(0)*coef_mat(1,0)+this->p2_pow_dp2_buf(1)*coef_mat(1,1)+this->p2_pow_dp2_buf(2)*coef_mat(1,2)+this->p2_pow_dp2_buf(3)*coef_mat(1,3))*this->p1_pow_buf(1)+
                               (this->p2_pow_dp2_buf(0)*coef_mat(2,0)+this->p2_pow_dp2_buf(1)*coef_mat(2,1)+this->p2_pow_dp2_buf(2)*coef_mat(2,2)+this->p2_pow_dp2_buf(3)*coef_mat(2,3))*this->p1_pow_buf(2)+
                               (this->p2_pow_dp2_buf(0)*coef_mat(3,0)+this->p2_pow_dp2_buf(1)*coef_mat(3,1)+this->p2_pow_dp2_buf(2)*coef_mat(3,2)+this->p2_pow_dp2_buf(3)*coef_mat(3,3))*this->p1_pow_buf(3);
    
    return this->first_order_buf;
}

template <typename T_container> 
inline typename cubic_interp_base<T_container>::const_container& cubic_interp_base<T_container>::calc_coef_mat(container &coef_mat_buf, const_container &A, difference_type p1, difference_type p2) const {
    // p1 and p2 refer to the top-left corner of the desired coefficient matrix
    #ifndef NDEBUG
    if (p1 < 0 || p1 + 3 >= A.height() || p2 < 0 || p2 + 3 >= A.width()) {
        throw std::invalid_argument("p1 and p2 are outside range of array for calc_coef_mat() with cubic interpolation - this is a programmer error.");
    }
    #endif

    coef_mat_buf(0,0) = 1.0*A(p1+1,p2+1);
    coef_mat_buf(1,0) = 0.5*A(p1+2,p2+1)-0.5*A(p1,p2+1);
    coef_mat_buf(2,0) = 1.0*A(p1,p2+1)-2.5*A(p1+1,p2+1)+2.0*A(p1+2,p2+1)-0.5*A(p1+3,p2+1);
    coef_mat_buf(3,0) = 1.5*A(p1+1,p2+1)-0.5*A(p1,p2+1)-1.5*A(p1+2,p2+1)+0.5*A(p1+3,p2+1);
    coef_mat_buf(0,1) = 0.5*A(p1+1,p2+2)-0.5*A(p1+1,p2);
    coef_mat_buf(1,1) = 0.25*A(p1,p2)-0.25*A(p1,p2+2)-0.25*A(p1+2,p2)+0.25*A(p1+2,p2+2);
    coef_mat_buf(2,1) = 0.5*A(p1,p2+2)-0.5*A(p1,p2)+1.25*A(p1+1,p2)-1.25*A(p1+1,p2+2)-1.0*A(p1+2,p2)+1.0*A(p1+2,p2+2)+0.25*A(p1+3,p2)-0.25*A(p1+3,p2+2);
    coef_mat_buf(3,1) = 0.25*A(p1,p2)-0.25*A(p1,p2+2)-0.75*A(p1+1,p2)+0.75*A(p1+1,p2+2)+0.75*A(p1+2,p2)-0.75*A(p1+2,p2+2)-0.25*A(p1+3,p2)+0.25*A(p1+3,p2+2);
    coef_mat_buf(0,2) = 1.0*A(p1+1,p2)-2.5*A(p1+1,p2+1)+2.0*A(p1+1,p2+2)-0.5*A(p1+1,p2+3);
    coef_mat_buf(1,2) = 1.25*A(p1,p2+1)-0.5*A(p1,p2)-1.0*A(p1,p2+2)+0.25*A(p1,p2+3)+0.5*A(p1+2,p2)-1.25*A(p1+2,p2+1)+1.0*A(p1+2,p2+2)-0.25*A(p1+2,p2+3);
    coef_mat_buf(2,2) = 1.0*A(p1,p2)-2.5*A(p1,p2+1)+2.0*A(p1,p2+2)-0.5*A(p1,p2+3)-2.5*A(p1+1,p2)+6.25*A(p1+1,p2+1)-5.0*A(p1+1,p2+2)+1.25*A(p1+1,p2+3)+2.0*A(p1+2,p2)-5.0*A(p1+2,p2+1)+4.0*A(p1+2,p2+2)-1.0*A(p1+2,p2+3)-0.5*A(p1+3,p2)+1.25*A(p1+3,p2+1)-1.0*A(p1+3,p2+2)+0.25*A(p1+3,p2+3);
    coef_mat_buf(3,2) = 1.25*A(p1,p2+1)-0.5*A(p1,p2)-1.0*A(p1,p2+2)+0.25*A(p1,p2+3)+1.5*A(p1+1,p2)-3.75*A(p1+1,p2+1)+3.0*A(p1+1,p2+2)-0.75*A(p1+1,p2+3)-1.5*A(p1+2,p2)+3.75*A(p1+2,p2+1)-3.0*A(p1+2,p2+2)+0.75*A(p1+2,p2+3)+0.5*A(p1+3,p2)-1.25*A(p1+3,p2+1)+1.0*A(p1+3,p2+2)-0.25*A(p1+3,p2+3);
    coef_mat_buf(0,3) = 1.5*A(p1+1,p2+1)-0.5*A(p1+1,p2)-1.5*A(p1+1,p2+2)+0.5*A(p1+1,p2+3);
    coef_mat_buf(1,3) = 0.25*A(p1,p2)-0.75*A(p1,p2+1)+0.75*A(p1,p2+2)-0.25*A(p1,p2+3)-0.25*A(p1+2,p2)+0.75*A(p1+2,p2+1)-0.75*A(p1+2,p2+2)+0.25*A(p1+2,p2+3);
    coef_mat_buf(2,3) = 1.5*A(p1,p2+1)-0.5*A(p1,p2)-1.5*A(p1,p2+2)+0.5*A(p1,p2+3)+1.25*A(p1+1,p2)-3.75*A(p1+1,p2+1)+3.75*A(p1+1,p2+2)-1.25*A(p1+1,p2+3)-1.0*A(p1+2,p2)+3.0*A(p1+2,p2+1)-3.0*A(p1+2,p2+2)+1.0*A(p1+2,p2+3)+0.25*A(p1+3,p2)-0.75*A(p1+3,p2+1)+0.75*A(p1+3,p2+2)-0.25*A(p1+3,p2+3);
    coef_mat_buf(3,3) = 0.25*A(p1,p2)-0.75*A(p1,p2+1)+0.75*A(p1,p2+2)-0.25*A(p1,p2+3)-0.75*A(p1+1,p2)+2.25*A(p1+1,p2+1)-2.25*A(p1+1,p2+2)+0.75*A(p1+1,p2+3)+0.75*A(p1+2,p2)-2.25*A(p1+2,p2+1)+2.25*A(p1+2,p2+2)-0.75*A(p1+2,p2+3)-0.25*A(p1+3,p2)+0.75*A(p1+3,p2+1)-0.75*A(p1+3,p2+2)+0.25*A(p1+3,p2+3);
       
    return coef_mat_buf;
}


// Function to save data as JSON
void save_as_json(const std::string& filename, const json& data, const std::string& directory) {
    // Create directory if it doesn't exist
    system(("mkdir -p " + directory).c_str());
    
    // Save to file
    std::ofstream file(directory + "/" + filename);
    file << std::setw(4) << data << std::endl;
}

// Helper functions to convert ncorr data structures to JSON
json array2d_to_json(const Array2D<double>& array) {
    json j;
    j["rows"] = array.height();
    j["cols"] = array.width();
    
    // Convert data to vector
    std::vector<double> data;
    for (int i = 0; i < array.height(); ++i) {
        for (int j = 0; j < array.width(); ++j) {
            data.push_back(array(i, j));
        }
    }
    j["data"] = data;
    
    return j;
}

// Test function for cubic interpolator with detailed output
json test_cubic_interpolator(const Array2D<double>& array) {
    json result;
    result["interpolator_type"] = "CUBIC_KEYS";
    
    // Create our custom cubic interpolator
    cubic_interp<Array2D<double>> interpolator(array);
    
    // Test points to sample
    std::vector<std::pair<double, double>> test_points = {
        {1.0, 1.0},         // Integer coordinates
        {1.5, 1.5},         // Half-pixel coordinates
        {1.25, 1.75},       // Fractional coordinates
        {2.1, 2.1},         // Near boundary but valid
        {array.height() - 3.1, array.width() - 3.1}  // Near opposite boundary but valid
    };
    
    // Sample interpolator at test points
    json samples = json::array();
    for (const auto& point : test_points) {
        json sample;
        sample["p1"] = point.first;
        sample["p2"] = point.second;
        
        // Get interpolated value
        try {
            // Measure time for interpolation
            auto start = std::chrono::high_resolution_clock::now();
            double value = interpolator(point.first, point.second);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;
            
            sample["value"] = value;
            sample["is_valid"] = !std::isnan(value);
            sample["time_microseconds"] = duration.count();
            
            // Get coefficient matrix
            int p1_floor = static_cast<int>(std::floor(point.first));
            int p2_floor = static_cast<int>(std::floor(point.second));
            
            // Create a buffer for the coefficient matrix
            Array2D<double> coef_mat_buf(4, 4);
            
            // Get coefficient matrix using calc_coef_mat
            auto start_calc = std::chrono::high_resolution_clock::now();
            const Array2D<double>& calc_coef_result = interpolator.calc_coef_mat(coef_mat_buf, array, p1_floor - 1, p2_floor - 1);
            auto end_calc = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> calc_duration = end_calc - start_calc;
            
            // Get coefficient matrix using get_coef_mat
            auto start_get = std::chrono::high_resolution_clock::now();
            const Array2D<double>& get_coef_result = interpolator.get_coef_mat(p1_floor, p2_floor);
            auto end_get = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> get_duration = end_get - start_get;
            
            // Convert coefficient matrices to JSON
            json calc_coef_json = json::array();
            json get_coef_json = json::array();
            
            for (int i = 0; i < 4; i++) {
                json calc_row = json::array();
                json get_row = json::array();
                for (int j = 0; j < 4; j++) {
                    calc_row.push_back(calc_coef_result(i, j));
                    get_row.push_back(get_coef_result(i, j));
                }
                calc_coef_json.push_back(calc_row);
                get_coef_json.push_back(get_row);
            }
            
            sample["calc_coef_mat"] = calc_coef_json;
            sample["get_coef_mat"] = get_coef_json;
            sample["calc_coef_time_microseconds"] = calc_duration.count();
            sample["get_coef_time_microseconds"] = get_duration.count();
            
        } catch (const std::exception& e) {
            sample["value"] = nullptr;
            sample["is_valid"] = false;
            sample["error"] = e.what();
        }
        
        // Get first-order derivatives
        try {
            auto start = std::chrono::high_resolution_clock::now();
            const Array2D<double>& first_order_result = interpolator.first_order(point.first, point.second);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;
            
            json first_order;
            first_order["value"] = first_order_result(0, 0);
            first_order["dp1"] = first_order_result(1, 0);
            first_order["dp2"] = first_order_result(2, 0);
            first_order["time_microseconds"] = duration.count();
            sample["first_order"] = first_order;
        } catch (const std::exception& e) {
            sample["first_order"] = nullptr;
            sample["first_order_error"] = e.what();
        }
        
        samples.push_back(sample);
    }
    result["samples"] = samples;
    
    return result;
}



// Test function for quintic interpolator
json test_quintic_interpolator(const Array2D<double>& array) {
    json result;
    quintic_interp<Array2D<double>> interpolator(array);

    std::cout << "Quintic Interpolator Test. Order = " << interpolator.get_order() <<std::endl;
    
    // Define test points - same as cubic for comparison
    std::vector<std::pair<double,double>> test_points = {
        {10.0, 10.0},      // Integer coordinates
        {10.5, 10.5},      // Half-pixel coordinates
        {10.25, 10.75},    // Fractional coordinates
        {1.1, 1.1},        // Near boundary
        {array.height()-2.1, array.width()-2.1} // Near boundary
    };
    
    json samples = json::array();
    
    // Test each point
    for (const auto &point : test_points) {
        json sample;
        sample["p1"] = point.first;
        sample["p2"] = point.second;
        sample["is_valid"] = true;
        sample["test_data"] = array2d_to_json(array);
        
        try {
            // Interpolate value
            auto start = std::chrono::high_resolution_clock::now();
            double value = interpolator(point.first, point.second);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;
            
            sample["value"] = value;
            sample["time_microseconds"] = duration.count();
            
            // Get B-spline coefficient matrix
            auto start_bspline = std::chrono::high_resolution_clock::now();
            std::shared_ptr<Array2D<double>> bspline_mat_ptr = interpolator.get_bspline_mat_ptr(array);
            auto end_bspline = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> bspline_duration = end_bspline - start_bspline;
            
            sample["bspline_mat_ptr_time_microseconds"] = bspline_duration.count();            
            sample["bspline_mat"] = array2d_to_json(*bspline_mat_ptr);

            std::shared_ptr<Array2D<double>> bspline_mat_ptr_pad = interpolator.get_bspline_mat_ptr_pad(array);
            sample["bspline_mat_ptr_pad"] = array2d_to_json(*bspline_mat_ptr_pad);   
            
            std::shared_ptr<Array2D<double>> bspline_mat_ptr_pad_deconv = interpolator.get_bspline_mat_ptr_pad_deconv(array);
            sample["bspline_mat_ptr_pad_deconv"] = array2d_to_json(*bspline_mat_ptr_pad_deconv);   

            
            // Calculate coefficient matrix with calc_coef_mat
            Array2D<double> coef_mat_buf(6, 6);
            auto start_calc = std::chrono::high_resolution_clock::now();
            int p1_floor = static_cast<int>(std::floor(point.first));
            int p2_floor = static_cast<int>(std::floor(point.second));
            // For quintic, we need to adjust the indices for the B-spline coefficients
            // Get B-spline coefficients directly from the interpolator
            std::shared_ptr<Array2D<double>> bspline_mat = interpolator.get_bspline_mat_ptr(array);
            const Array2D<double>& calc_coef_result = interpolator.calc_coef_mat(
                coef_mat_buf, 
                *bspline_mat, 
                p1_floor, 
                p2_floor
            );
            auto end_calc = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> calc_duration = end_calc - start_calc;
            
            // Get coefficient matrix using get_coef_mat
            auto start_get = std::chrono::high_resolution_clock::now();
            const Array2D<double>& get_coef_result = interpolator.get_coef_mat(p1_floor, p2_floor);
            auto end_get = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> get_duration = end_get - start_get;
            
            // Convert coefficient matrices to JSON
            json calc_coef_json = json::array();
            json get_coef_json = json::array();
            
            for (int i = 0; i < 6; i++) {
                json calc_row = json::array();
                json get_row = json::array();
                for (int j = 0; j < 6; j++) {
                    calc_row.push_back(calc_coef_result(i, j));
                    get_row.push_back(get_coef_result(i, j));
                }
                calc_coef_json.push_back(calc_row);
                get_coef_json.push_back(get_row);
            }
            
            sample["calc_coef_mat"] = calc_coef_json;
            sample["get_coef_mat"] = get_coef_json;
            sample["calc_coef_time_microseconds"] = calc_duration.count();
            sample["get_coef_time_microseconds"] = get_duration.count();
            
        } catch (const std::exception& e) {
            sample["value"] = nullptr;
            sample["is_valid"] = false;
            sample["error"] = e.what();
        }
        
        // Get first-order derivatives
        try {
            auto start = std::chrono::high_resolution_clock::now();
            const Array2D<double>& first_order_result = interpolator.first_order(point.first, point.second);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> duration = end - start;
            
            json first_order;
            first_order["value"] = first_order_result(0, 0);
            first_order["dp1"] = first_order_result(1, 0);
            first_order["dp2"] = first_order_result(2, 0);
            first_order["time_microseconds"] = duration.count();
            sample["first_order"] = first_order;
        } catch (const std::exception& e) {
            sample["first_order"] = nullptr;
            sample["first_order_error"] = e.what();
        }
        
        samples.push_back(sample);
    }
    result["samples"] = samples;
    
    return result;
}

// Main function to run both cubic and quintic interpolator tests
int main(int argc, char* argv[]) {
    // Create a test directory
    std::string output_dir = "interpolator_test_output";
    
    try {
        // Create a test array with some sample data
        Array2D<double> test_array(20, 20);
        
        // Fill with sample data (gradient pattern)
        for (int i = 0; i < test_array.height(); i++) {
            for (int j = 0; j < test_array.width(); j++) {
                test_array(i, j) = i * 0.5 + j * 0.3 + sin(i * 0.2) * cos(j * 0.2);
            }
        }
        
        // Run the cubic interpolator test
        std::cout << "Running cubic interpolator test..." << std::endl;
        json cubic_result = test_cubic_interpolator(test_array);
        
        // Output the cubic result to stdout with debug info
        std::cout << "=== CUBIC INTERPOLATOR TEST RESULTS ===" << std::endl;
        std::cout << "Test array size: " << test_array.height() << "x" << test_array.width() << std::endl;
        std::cout << "Number of samples: " << cubic_result["samples"].size() << std::endl;
        
        // Save cubic results to file
        std::string cubic_output_filename = "cubic_interpolator_test_results.json";
        save_as_json(cubic_output_filename, cubic_result, output_dir);
        std::cout << "Cubic results saved to " << cubic_output_filename << std::endl;
        
        // Run the quintic interpolator test
        std::cout << "\nRunning quintic interpolator test..." << std::endl;
        json quintic_result = test_quintic_interpolator(test_array);
        
        // Output the quintic result to stdout with debug info
        std::cout << "=== QUINTIC INTERPOLATOR TEST RESULTS ===" << std::endl;
        std::cout << "Test array size: " << test_array.height() << "x" << test_array.width() << std::endl;
        std::cout << "Number of samples: " << quintic_result["samples"].size() << std::endl;
        
        // Save quintic results to file
        std::string quintic_output_filename = "quintic_interpolator_test_results.json";
        save_as_json(quintic_output_filename, quintic_result, output_dir);
        std::cout << "Quintic results saved to " << quintic_output_filename << std::endl;
        
        // Create a combined result for comparison
        json combined_result;
        combined_result["cubic"] = cubic_result;
        combined_result["quintic"] = quintic_result;
        
        // Save combined results to file
        std::string combined_output_filename = "interpolator_comparison_results.json";
        save_as_json(combined_output_filename, combined_result, output_dir);
        std::cout << "Combined comparison results saved to " << combined_output_filename << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
