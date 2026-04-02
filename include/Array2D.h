/* 
 * File:   Array2D.h
 * Author: justin
 *
 * Created on December 30, 2014, 4:55 PM
 */

#ifndef ARRAY2D_H
#define	ARRAY2D_H

#include <cstddef>
#include <memory>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <utility>
#include <mutex>                
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

namespace ncorr {          

namespace details {       
    class last_index { };
    class all_range { };
    
    template <typename> class base_iterator;
    template <typename> class simple_iterator;
    template <typename> class sub_iterator;
    template <typename> class bool_iterator;
    template <typename> class interface_iterator;
    
    template <typename> class base_region;
    template <typename> class simple_region;
    template <typename> class sub_region;
    template <typename> class bool_region;
    template <typename> class interface_region;  
    
    template <typename> class base_interp;
    template <typename> class nearest_interp;
    template <typename> class linear_interp;
    template <typename> class cubic_interp;
    template <typename> class cubic_interp_precompute;
    template <typename> class quintic_interp;
    template <typename> class quintic_interp_precompute;
    template <typename> class interface_interp;      
        
    template <typename> class base_linsolver;
    template <typename> class LU_linsolver;
    template <typename> class QR_linsolver;
    template <typename> class CHOL_linsolver;
    template <typename> class interface_linsolver;   
}

// These can be passed as arguments in Array2D indexing operations -----------//
// "last" is used instead of "end" because end() is the function call for the 
// end iterator and results in a name collision when used in Array2D methods.
const class details::last_index last; 
const class details::all_range all;

// Define enums for functions that take them ---------------------------------//
enum class PAD { ZEROS, EXPAND_EDGES };
// Make sure ordering of INTERP depends on polynomial order, as some functions
// only work for higher order interpolation and use relational operators on 
// input INTERP to test for this.
enum class INTERP { NEAREST, LINEAR, CUBIC_KEYS, CUBIC_KEYS_PRECOMPUTE, QUINTIC_BSPLINE, QUINTIC_BSPLINE_PRECOMPUTE }; 
enum class LINSOLVER { LU, QR, CHOL };

template <typename T, typename T_alloc>
class Array2D;

Array2D<double, std::allocator<double>> conv(const Array2D<double, std::allocator<double>> &A,
                                             const Array2D<double, std::allocator<double>> &B);
Array2D<double, std::allocator<double>> deconv(const Array2D<double, std::allocator<double>> &A,
                                               const Array2D<double, std::allocator<double>> &B);
Array2D<double, std::allocator<double>> xcorr(const Array2D<double, std::allocator<double>> &A,
                                              const Array2D<double, std::allocator<double>> &B);

namespace details {
Array2D<double, std::allocator<double>> blas_mat_mult(const Array2D<double, std::allocator<double>> &A,
                                                      const Array2D<double, std::allocator<double>> &B);
}

template <typename T, typename T_alloc = std::allocator<T>> 
class Array2D final { 
// -------------------------------------------------------------------------- // 
// -------------------------------------------------------------------------- // 
// Note: This container only works with stateless, default-constructed        //
// allocators. Allocators are not passed around; they are created when        //
// Array2Ds are created, using their default constructor.                     //
// -------------------------------------------------------------------------- //  
// -------------------------------------------------------------------------- // 
    public:                
        typedef T                                                                      value_type;
        typedef T*                                                                        pointer; 
        typedef const T*                                                            const_pointer; 
        typedef T&                                                                      reference;  
        typedef const T&                                                          const_reference;   
        typedef std::size_t                                                             size_type; 
        typedef std::ptrdiff_t                                                    difference_type;    
        typedef std::pair<difference_type,difference_type>                                 coords;   
        typedef details::interface_iterator<details::base_iterator<Array2D>>             iterator;  
        typedef details::interface_iterator<details::base_iterator<const Array2D>> const_iterator;
        typedef details::interface_region<details::base_region<Array2D>>                   region;  
        typedef details::interface_region<details::base_region<const Array2D>>       const_region; 
        typedef details::interface_interp<details::base_interp<Array2D>>             interpolator; 
        typedef details::interface_linsolver<details::base_linsolver<Array2D>>          linsolver; 
        typedef T_alloc                                                            allocator_type;
        typedef std::allocator_traits<allocator_type>                        allocator_traits_type;
        typedef Array2D                                                                 container;
        typedef const Array2D                                                     const_container;
        template <typename T_other>
        using rebind_allocator = typename allocator_traits_type::template rebind_alloc<T_other>;
        typedef rebind_allocator<bool>                                            bool_allocator;
        typedef Array2D<bool, bool_allocator>                                      bool_container;
                                        
        template <typename T2, typename T_alloc2>
        friend class Array2D;
                
        // Rule of 5 and destructor ------------------------------------------//        
        // Array2D has value-like semantics
        Array2D() noexcept : h(), w(), s(), ptr(nullptr) { }
        Array2D(const Array2D &A) : h(A.h), w(A.w), s(A.s), ptr(allocate_and_copy(A.s, A.ptr)) { }
        Array2D(Array2D &&A) noexcept : h(A.h), w(A.w), s(A.s), ptr(A.ptr) { A.make_null(); }
        Array2D& operator=(const Array2D&);
        Array2D& operator=(Array2D&&) noexcept; 
        ~Array2D() noexcept { destroy_and_deallocate(); }
        
        // Additional Constructors -------------------------------------------//   
        Array2D(difference_type, difference_type, const_reference = value_type());
        Array2D(std::initializer_list<value_type>);
        Array2D(std::initializer_list<std::initializer_list<value_type>>);
        // Allow conversions from other array types only explicitly
        template <typename T2, typename T_alloc2> 
        explicit Array2D(const Array2D<T2,T_alloc2> &A) : h(A.h), w(A.w), s(A.s), ptr(allocate_and_copy(A.s, A.ptr)) { }
        // Allow implicit conversions from regions - overloads for region and 
        // const_region are both needed because implicit conversions do not chain. 
        Array2D(const const_region &reg) : h(reg.region_height()), w(reg.region_width()), s(reg.region_size()), ptr(allocate_and_copy(reg.region_size(), reg.begin())) { } 
        Array2D(const region &reg) : h(reg.region_height()), w(reg.region_width()), s(reg.region_size()), ptr(allocate_and_copy(reg.region_size(), reg.begin())) { }
        
        // Static factory methods --------------------------------------------//
        // Allow Array2Ds to be formed from binary filename or a stream. Only 
        // allow for arithmetic types.
        template <typename T_output = Array2D>        
        static typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type load(const std::string&);
        template <typename T_output = Array2D>     
        static typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type load(std::ifstream&);
        
        // Conversion --------------------------------------------------------//
        // Only allow conversions explicitly when array is 1x1
        explicit operator value_type();
        
        //--------------------------------------------------------------------//
        // Single element indexing -------------------------------------------//
        //--------------------------------------------------------------------//
        
        // Linear indexing
        const_reference operator()(difference_type) const;      
        reference operator()(difference_type p) { return const_cast<reference>((*const_cast<const Array2D*>(this))(p)); }
        // "last" overload
        const_reference operator()(const details::last_index&) const { return (*this)(s - 1); }   
        reference operator()(const details::last_index &last) { return const_cast<reference>((*const_cast<const Array2D*>(this))(last)); }

        // 2D indexing
        const_reference operator()(difference_type, difference_type) const;
        reference operator()(difference_type p1, difference_type p2) { return const_cast<reference>((*const_cast<const Array2D*>(this))(p1,p2)); } 
        // "last" overloads
        const_reference operator()(difference_type p1, const details::last_index&) const { return (*this)(p1,w - 1); } 
        reference operator()(difference_type p1, const details::last_index &last) { return const_cast<reference>((*const_cast<const Array2D*>(this))(p1,last)); } 
        const_reference operator()(const details::last_index&, difference_type p2) const { return (*this)(h - 1,p2); } 
        reference operator()(const details::last_index &last, difference_type p2) { return const_cast<reference>((*const_cast<const Array2D*>(this))(last,p2)); } 
        const_reference operator()(const details::last_index&, const details::last_index&) const { return (*this)(h - 1,w - 1); } 
        reference operator()(const details::last_index &last1, const details::last_index &last2) { return const_cast<reference>((*const_cast<const Array2D*>(this))(last1,last2)); } 
            
        //--------------------------------------------------------------------//
        // Regions -----------------------------------------------------------//
        //--------------------------------------------------------------------//
        // Note that regions that use "ranges" use inclusive bounds ONLY in the
        // interface. Empty ranges are supported by having the second index less 
        // than the first index. These ranges are converted by get_range() such 
        // that the end range is non-inclusive for computation. In this case
        // an empty range is denoted by two indices of the same value.
    private:        
        struct r_convert {
            // Five overall supported conversions to range:
            enum class RANGE { SINGLE, LAST, COORDS, LAST_COORDS, ALL } range_type;
            r_convert(RANGE range_type, difference_type p1 = 0, difference_type p2 = 0) : range_type(range_type), p1(p1), p2(p2) { }
            difference_type p1;
            difference_type p2;
        };
        // Derived classes of r_convert allows get_range() function to determine 
        // which value to substitute "last" and "all" with - either s, h, or w.
        struct r_convert_1D final : r_convert {
            // Three types of conversions for 1D
            r_convert_1D(difference_type p1, difference_type p2) : r_convert(r_convert::RANGE::COORDS, p1, p2) { }
            r_convert_1D(difference_type p1, const details::last_index&) : r_convert(r_convert::RANGE::LAST_COORDS, p1) { }
            r_convert_1D(const details::all_range&) : r_convert(r_convert::RANGE::ALL) { }
        };
        struct r_convert_2D_1 final : r_convert {
            // Five types of conversions for 2D first argument
            r_convert_2D_1(difference_type p) : r_convert(r_convert::RANGE::SINGLE,p) { }
            r_convert_2D_1(const details::last_index&) : r_convert(r_convert::RANGE::LAST) { }
            r_convert_2D_1(difference_type p1, difference_type p2) : r_convert(r_convert::RANGE::COORDS, p1, p2) { }
            r_convert_2D_1(difference_type p1, const details::last_index&) : r_convert(r_convert::RANGE::LAST_COORDS, p1) { }
            r_convert_2D_1(const details::all_range&) : r_convert(r_convert::RANGE::ALL) { }
        };
        struct r_convert_2D_2 final : r_convert {
            // Five types of conversions for 2D second argument
            r_convert_2D_2(difference_type p) : r_convert(r_convert::RANGE::SINGLE,p) { }
            r_convert_2D_2(const details::last_index&) : r_convert(r_convert::RANGE::LAST) { }
            r_convert_2D_2(difference_type p1, difference_type p2) : r_convert(r_convert::RANGE::COORDS, p1, p2) { }
            r_convert_2D_2(difference_type p1, const details::last_index&) : r_convert(r_convert::RANGE::LAST_COORDS, p1) { }
            r_convert_2D_2(const details::all_range&) : r_convert(r_convert::RANGE::ALL) { }
        };  
        
    public:
        // 1D Sub Array Indexing ---------------------------------------------//
        // Special empty index (i.e. A()) will get all elements and treat it 
        // like a 1D range over the entire array.
        const_region operator()() const { return const_region(new details::simple_region<const Array2D>(*this,{0,s})); }
        region operator()() { return region(new details::simple_region<Array2D>(*this,{0,s})); }
        
        const_region operator()(const r_convert_1D &r_sub_1D) const { return const_region(new details::simple_region<const Array2D>(*this,get_range(r_sub_1D))); }
        region operator()(const r_convert_1D &r_sub_1D) { return region(new details::simple_region<Array2D>(*this,get_range(r_sub_1D))); }
        
        // 2D Sub Array Indexing ---------------------------------------------// 
        const_region operator()(const r_convert_2D_1 &r_sub1_2D,const r_convert_2D_2 &r_sub2_2D) const { return const_region(new details::sub_region<const Array2D>(*this,get_range(r_sub1_2D),get_range(r_sub2_2D))); }
        region operator()(const r_convert_2D_1 &r_sub1_2D, const r_convert_2D_2 &r_sub2_2D) { return region(new details::sub_region<Array2D>(*this,get_range(r_sub1_2D),get_range(r_sub2_2D))); }
        
        // Logical indexing --------------------------------------------------//
        // Note that region will form a copy of the input bool_container. This is 
        // templated to prevent ambiguous function calls with 1D range indexing
        // since Array2D has a constructor which takes two integers; this 
        // result in an ambiguous function call when using the {,} notation. 
        template <typename T_container>
        typename std::enable_if<std::is_same<typename T_container::container,bool_container>::value, const_region>::type operator()(T_container A) const { return const_region(new details::bool_region<const Array2D>(*this,std::move(A))); }
        template <typename T_container>
        typename std::enable_if<std::is_same<typename T_container::container,bool_container>::value, region>::type operator()(T_container A) { return region(new details::bool_region<Array2D>(*this,std::move(A))); }
                
        // Provide overload for regions since implicit conversions cant be done 
        // with template methods.
        template <typename T_container>
        typename std::enable_if<std::is_same<typename T_container::container,bool_container>::value, const_region>::type operator()(const details::interface_region<details::base_region<T_container>> &reg) const { return const_region(new details::bool_region<const Array2D>(*this,reg)); }
        template <typename T_container>
        typename std::enable_if<std::is_same<typename T_container::container,bool_container>::value, region>::type operator()(const details::interface_region<details::base_region<T_container>> &reg) { return region(new details::bool_region<Array2D>(*this,reg)); }
                    
        // Iterators ---------------------------------------------------------//
        // Use simple iterators; these will iterate over the entire array.
        iterator begin() { return iterator(new details::simple_iterator<Array2D>(*this,0)); }
        iterator end() { return iterator(new details::simple_iterator<Array2D>(*this,s)); }
        const_iterator begin() const { return const_iterator(new details::simple_iterator<const Array2D>(*this,0)); }
        const_iterator end() const { return const_iterator(new details::simple_iterator<const Array2D>(*this,s)); }
        const_iterator cbegin() const { return const_iterator(new details::simple_iterator<const Array2D>(*this,0)); }
        const_iterator cend() const { return const_iterator(new details::simple_iterator<const Array2D>(*this,s)); }
        
        // Interpolator ------------------------------------------------------//
        template<typename T_output = interpolator> 
        typename std::enable_if<std::is_floating_point<value_type>::value, T_output>::type get_interpolator(INTERP interp_type) const { 
            interpolator interp;
            switch (interp_type) {             
                case INTERP::NEAREST : interp = interpolator(new details::nearest_interp<Array2D>(*this)); break; 
                case INTERP::LINEAR : interp = interpolator(new details::linear_interp<Array2D>(*this)); break;      
                case INTERP::CUBIC_KEYS : interp = interpolator(new details::cubic_interp<Array2D>(*this)); break;     
                case INTERP::CUBIC_KEYS_PRECOMPUTE : interp = interpolator(new details::cubic_interp_precompute<Array2D>(*this)); break;         
                case INTERP::QUINTIC_BSPLINE : interp = interpolator(new details::quintic_interp<Array2D>(*this)); break;      
                case INTERP::QUINTIC_BSPLINE_PRECOMPUTE : interp = interpolator(new details::quintic_interp_precompute<Array2D>(*this)); break;       
            } 
            
            return interp;
        }
                
        // Linsolvers --------------------------------------------------------//
        template<typename T_output = linsolver> 
        typename std::enable_if<std::is_floating_point<value_type>::value, T_output>::type get_linsolver(LINSOLVER linsolver_type) const;
                
        //--------------------------------------------------------------------//
        // Operations interface ----------------------------------------------//
        //--------------------------------------------------------------------//
        // These operations are special in that they allow for implicit       //
        // conversions with Array2D, so they can be used with regions (look   //
        // at item 46 in effective C++).                                      // 
        // Overloads are only provided for R-value types to provide some      //
        // performance improvements. If regions are passed, they will trigger //
        // the R-value overload if one exists.                                //
        // These functions call private template member functions that use    //
        // SFINAE to allow some operations for only some types.               //
        // can only be called for arithmetic types).                          //
        // -------------------------------------------------------------------//
              
        // General operations ------------------------------------------------//      
        template <typename T2> 
        friend Array2D<T2, rebind_allocator<T2>> convert(const Array2D &A, const T2&) { return Array2D<T2, rebind_allocator<T2>>(A); }
        friend std::ostream& operator<<(std::ostream &os, const Array2D &A) { return A.this_stream(os); }      
        friend Array2D repmat(const Array2D &A, difference_type rows, difference_type cols) { return A.this_repmat(rows,cols); }    
        friend Array2D pad(const Array2D &A, difference_type padding, PAD pad_type = PAD::ZEROS) { return A.this_pad(padding,pad_type); }    
        friend Array2D t(const Array2D &A) { return A.this_t(); }  
        friend void save(const Array2D &A, const std::string &filename) { A.this_save(filename); }
        friend void save(const Array2D &A, std::ofstream &os) { A.this_save(os); }
                
        // Logical operators -------------------------------------------------//        
        friend bool isequal(const Array2D &A, const Array2D &B) { return A.this_isequal(B); }    
        friend bool_container operator==(const Array2D &A, const Array2D &B) { return A.this_equals(B); }          
        friend bool_container operator==(const Array2D &A, const_reference val) { return A.this_equals(val); }
        friend bool_container operator==(const_reference val, const Array2D &A) { return A.this_equals(val); }      
        friend bool_container operator!=(const Array2D &A, const Array2D &B) { return A.this_notequals(B); }       
        friend bool_container operator!=(const Array2D &A, const_reference val) { return A.this_notequals(val); }
        friend bool_container operator!=(const_reference val, const Array2D &A) { return A.this_notequals(val); }       
        friend bool_container operator&(const Array2D &A, const Array2D &B) { return A.this_and(B); }     
        friend bool_container operator|(const Array2D &A, const Array2D &B) { return A.this_or(B); }   
        friend bool_container operator~(const Array2D &A) { return A.this_negate(); } 
        friend bool any_true(const Array2D &A) { return A.this_any_true(); }        
        friend bool all_true(const Array2D &A) { return A.this_all_true(); }
        friend difference_type find(const Array2D &A, difference_type start = 0) { return A.this_find(start); }
        
        // Relational Operators ----------------------------------------------//        
        friend bool_container operator>(const Array2D &A, const Array2D &B) { return A.this_greaterthan(B); }     
        friend bool_container operator>(const Array2D &A, const_reference val) { return A.this_greaterthan(val); }       
        friend bool_container operator>=(const Array2D &A, const Array2D &B) { return A.this_greaterthanorequalto(B); }   
        friend bool_container operator>=(const Array2D &A, const_reference val) { return A.this_greaterthanorequalto(val); }        
        friend bool_container operator<(const Array2D &A, const Array2D &B) { return A.this_lessthan(B); }    
        friend bool_container operator<(const Array2D &A, const_reference val) { return A.this_lessthan(val); }
        friend bool_container operator<=(const Array2D &A, const Array2D &B) { return A.this_lessthanorequalto(B); }  
        friend bool_container operator<=(const Array2D &A, const_reference val) { return A.this_lessthanorequalto(val); }  
        
        // Arithmetic operators ----------------------------------------------//                 
        friend Array2D operator+(const Array2D &A, const Array2D &B) { Array2D C(A); return C += B; }
        friend Array2D operator+(const Array2D &A, Array2D &&B) { return B += A; }
        friend Array2D operator+(Array2D &&A, const Array2D &B) { return A += B; }
        friend Array2D operator+(Array2D &&A, Array2D &&B) { return A += B; }

        friend Array2D operator+(const Array2D &A, const_reference val) { Array2D B(A); return B += val; }
        friend Array2D operator+(Array2D &&A, const_reference val) { return A += val; }
        friend Array2D operator+(const_reference val, const Array2D &A) { Array2D B(A); return B += val; }
        friend Array2D operator+(const_reference val, Array2D &&A) { return A += val; }
        
        // Subtraction is not commutative- so some wrangling is required for 
        // dealing with r-values
        friend Array2D operator-(const Array2D &A, const Array2D &B) { Array2D C(A); return C -= B; }
        friend Array2D operator-(const Array2D &A, Array2D &&B) { return B.this_flipsubassign(A); } // Cannot use -=
        friend Array2D operator-(Array2D &&A, const Array2D &B) { return A -= B; }
        friend Array2D operator-(Array2D &&A, Array2D &&B) { return A -= B; }

        friend Array2D operator-(const Array2D &A, const_reference val) { Array2D B(A); return B -= val; }
        friend Array2D operator-(Array2D &&A, const_reference val) { return A -= val; }
        friend Array2D operator-(const_reference val, const Array2D &A) { Array2D B(A); return B.this_flipsubassign(val); }
        friend Array2D operator-(const_reference val, Array2D &&A) { return A.this_flipsubassign(val); }
        
        // For element-wise multiplication of two arrays, use "mult" instead of 
        // operator* - this is reserved for matrix multiplication
        friend Array2D mult(const Array2D &A, const Array2D &B) { Array2D C(A); return C *= B; }
        friend Array2D mult(const Array2D &A, Array2D &&B) { return B *= A; }
        friend Array2D mult(Array2D &&A, const Array2D &B) { return A *= B; }
        friend Array2D mult(Array2D &&A, Array2D &&B) { return A *= B; }

        friend Array2D operator*(const Array2D &A, const_reference val) { Array2D B(A); return B *= val; }
        friend Array2D operator*(Array2D &&A, const_reference val) { return A *= val; }
        friend Array2D operator*(const_reference val, const Array2D &A) { Array2D B(A); return B *= val; }
        friend Array2D operator*(const_reference val, Array2D &&A) { return A *= val; }
                
        // Division is not commutative- so some wrangling is required for 
        // dealing with r-values
        friend Array2D operator/(const Array2D &A, const Array2D &B) { Array2D C(A); return C /= B; }
        friend Array2D operator/(const Array2D &A, Array2D &&B) { return B.this_flipdivassign(A); }
        friend Array2D operator/(Array2D &&A, const Array2D &B) { return A /= B; }
        friend Array2D operator/(Array2D &&A, Array2D &&B) { return A /= B; }

        friend Array2D operator/(const Array2D &A, const_reference val) { Array2D B(A); return B /= val; }
        friend Array2D operator/(Array2D &&A, const_reference val) { return A /= val; }
        friend Array2D operator/(const_reference val, const Array2D &A) { Array2D B(A); return B.this_flipdivassign(val); }
        friend Array2D operator/(const_reference val, Array2D &&A) { return A.this_flipdivassign(val); }
                                               
        friend Array2D sort(Array2D A) { return A.this_sort(); } // by-value   
        
        friend value_type sum(const Array2D &A) { return A.this_sum(); }
        
        friend value_type max(const Array2D &A) { return A.this_max(); }
        friend value_type min(const Array2D &A) { return A.this_min(); }
                
        friend value_type prctile(Array2D A, double percent) { return A.this_prctile(percent); } // by-value
        
        friend value_type median(Array2D A) { return prctile(std::move(A),0.5); } // by-value

        // Element-wise
        friend Array2D sqrt(Array2D A) { return A.this_sqrt(); } // by-value
        
        friend Array2D pow(Array2D A, double n) { return A.this_pow(n); } // by-value
        
        // Matrix multiplication and convolution operations ------------------//
        // These have special overloads for double type
        friend Array2D operator*(const Array2D &A, const Array2D &B) { return A.this_mat_mult(B); }     
        // Additional arithmetic operations ----------------------------------//
        friend value_type dot(const Array2D &x, const Array2D &y) { return x.this_dot(y); }     
        friend Array2D normalize(Array2D A) { return A.this_normalize(); } // by-value
        friend Array2D linsolve(const Array2D &A, const Array2D &b) { return A.this_linsolve(b); }
                               
        //--------------------------------------------------------------------//
        // End of operations interface ---------------------------------------//
        //--------------------------------------------------------------------//
        
        // Access ------------------------------------------------------------//
        difference_type height() const { return h; }
        difference_type width() const { return w; }
        difference_type size() const { return s; }
        pointer get_pointer() const { return ptr; }    
        
        // Utility -----------------------------------------------------------//
        bool empty() const { return s == 0; }
        template <typename T_container>
        bool same_size(const T_container &A) const { return h == A.h && w == A.w; }         
        difference_type sub2ind(difference_type p1, difference_type p2) const { return p1 + p2*h; }
        coords ind2sub(difference_type p) const { return {p % h, p / h}; }
        bool in_bounds(difference_type p) const { return p >= 0 && p < s; }
        bool in_bounds(difference_type p1, difference_type p2) const { return p1 >= 0 && p1 < h && p2 >= 0 && p2 < w; }
        std::string size_string() const { return std::to_string(s); }   
        std::string size_2D_string() const { return "(" + std::to_string(h) + "," + std::to_string(w) + ")"; }   
                
    private:          
        // -------------------------------------------------------------------//
        // These operations are private because the interface for operators   //
        // does not involve member functions. This is because all operators   //
        // are meant to work with Array2D AND regions, so operators need to   //
        // use implicit conversions on possibly all their arguments which is  //
        // not possible with member functions. Furthermore, these functions   //
        // can modify the array in-place, whereas interface functions never   //
        // modify in-place.                                                   //
        // -------------------------------------------------------------------//
        
        // General operation -------------------------------------------------//
        template<typename T_output = std::ostream&> 
        typename std::enable_if<std::is_arithmetic<value_type>::value, T_output>::type this_stream(std::ostream&) const;
        Array2D this_repmat(difference_type, difference_type) const; 
        Array2D this_pad(difference_type, PAD) const;
        Array2D this_t() const;
        template<typename T_output = void> 
        typename std::enable_if<std::is_arithmetic<value_type>::value, T_output>::type this_save(const std::string&) const;
        template<typename T_output = void> 
        typename std::enable_if<std::is_arithmetic<value_type>::value, T_output>::type this_save(std::ofstream&) const;
        
        // Logical operations ------------------------------------------------// 
        bool this_isequal(const Array2D&) const;  
        bool_container this_equals(const Array2D&) const;
        bool_container this_equals(const value_type&) const;        
        bool_container this_notequals(const Array2D&) const;
        bool_container this_notequals(const value_type&) const;
        bool_container this_and(const Array2D&) const;
        bool_container this_or(const Array2D&) const;                 
        bool_container this_negate() const;
        bool this_any_true() const;
        bool this_all_true() const;
        difference_type this_find(difference_type start) const;
        
        // Relational operations ---------------------------------------------//        
        bool_container this_greaterthan(const Array2D&) const;
        bool_container this_greaterthan(const value_type&) const;
        bool_container this_greaterthanorequalto(const Array2D&) const;
        bool_container this_greaterthanorequalto(const value_type&) const;
        bool_container this_lessthan(const Array2D&) const; 
        bool_container this_lessthan(const value_type&) const; 
        bool_container this_lessthanorequalto(const Array2D&) const;
        bool_container this_lessthanorequalto(const value_type&) const;
        
        // Arithmetic operations ---------------------------------------------//
        Array2D& operator+=(const Array2D&);
        Array2D& operator+=(const_reference);
        
        Array2D& operator-=(const Array2D&);
        Array2D& this_flipsubassign(const Array2D&);
        Array2D& operator-=(const_reference);
        Array2D& this_flipsubassign(const_reference);
        
        Array2D& operator*=(const Array2D&);
        Array2D& operator*=(const_reference);
        
        Array2D& operator/=(const Array2D&);
        Array2D& this_flipdivassign(const Array2D&);
        Array2D& operator/=(const_reference);      
        Array2D& this_flipdivassign(const_reference); 
        
        Array2D& operator+();
        Array2D& operator-();
                
        Array2D& this_sort();
        value_type this_sum() const;
        value_type this_max() const;
        value_type this_min() const;
        value_type this_prctile(double);
        
        // sqrt and pow use std::sqrt and std::pow which are only defined for arithmetic types
        template<typename T_output = Array2D&> 
        typename std::enable_if<std::is_arithmetic<value_type>::value, T_output>::type this_sqrt();
        template<typename T_output = Array2D&> 
        typename std::enable_if<std::is_arithmetic<value_type>::value, T_output>::type this_pow(double);
        
        // Provide specific overload for double
        template<typename T_output = Array2D> 
        typename std::enable_if<std::is_same<value_type,double>::value, T_output>::type this_mat_mult(const Array2D&) const;
        template<typename T_output = Array2D> 
        typename std::enable_if<!std::is_same<value_type,double>::value, T_output>::type this_mat_mult(const Array2D&) const;        
        
        // Additional arithmetic operations ----------------------------------//
        template<typename T_output = value_type> 
        typename std::enable_if<std::is_floating_point<value_type>::value, T_output>::type this_dot(const Array2D&) const;
        template<typename T_output = Array2D> 
        typename std::enable_if<std::is_floating_point<value_type>::value, T_output>::type this_normalize();
        template<typename T_output = Array2D> 
        typename std::enable_if<std::is_floating_point<value_type>::value, T_output>::type this_linsolve(const Array2D&) const;
        
        // Utility -----------------------------------------------------------//      
        coords get_range(const r_convert_1D&) const;
        coords get_range(const r_convert_2D_1&) const;
        coords get_range(const r_convert_2D_2&) const;     
        
        pointer allocate(difference_type);       
        pointer allocate_and_init(difference_type, const_reference);
        template <typename T_it> 
        pointer allocate_and_copy(difference_type, T_it);
        void destroy_and_deallocate();
        void make_null() { ptr = nullptr; h = w = s = 0; }
                    
        void chk_size_op(difference_type, difference_type, const std::string&) const;
        void chk_samesize_op(const Array2D&, const std::string&) const;
        void chk_minsize_op(difference_type, difference_type, const std::string&) const;
        void chk_column_op(const std::string&) const;
        void chk_square_op(const std::string&) const;    
        void chk_in_bounds_op(difference_type, const std::string&) const;   
        void chk_in_bounds_op(difference_type, difference_type, const std::string&) const;   
        void chk_mult_size(const Array2D&) const;    
        void chk_kernel_size(const Array2D&) const;
        
        difference_type h;
        difference_type w;
        difference_type s;
        pointer ptr;
        allocator_type alloc;
};
       
namespace details {        
    // Container traits will cause pointer, reference, iterator, region, etc to
    // be const if the T_container is const. Also includes const and nonconst
    // versions for convenience.
    template<typename T_container>
    struct container_traits {                
        typedef typename T_container::pointer                  nonconst_pointer; 
        typedef typename T_container::reference              nonconst_reference;  
        typedef typename T_container::iterator                nonconst_iterator;
        typedef typename T_container::region                    nonconst_region;
        typedef typename T_container::container              nonconst_container;
        
        typedef typename T_container::pointer                           pointer; 
        typedef typename T_container::reference                       reference;  
        typedef typename T_container::iterator                         iterator;
        typedef typename T_container::region                             region;
        typedef typename T_container::container                       container;
        
        typedef typename T_container::const_pointer               const_pointer; 
        typedef typename T_container::const_reference           const_reference;
        typedef typename T_container::const_iterator             const_iterator; 
        typedef typename T_container::const_region                 const_region;
        typedef typename T_container::const_container           const_container;
    };
    
    template<typename T_container> 
    struct container_traits<const T_container> { // Specialization for const container
        typedef typename T_container::pointer                  nonconst_pointer; 
        typedef typename T_container::reference              nonconst_reference;  
        typedef typename T_container::iterator                nonconst_iterator;
        typedef typename T_container::region                    nonconst_region;
        typedef typename T_container::container              nonconst_container;
        
        typedef typename T_container::const_pointer                     pointer; 
        typedef typename T_container::const_reference                 reference;  
        typedef typename T_container::const_iterator                   iterator;
        typedef typename T_container::const_region                       region;
        typedef typename T_container::const_container                 container;
        
        typedef typename T_container::const_pointer               const_pointer; 
        typedef typename T_container::const_reference           const_reference;
        typedef typename T_container::const_iterator             const_iterator; 
        typedef typename T_container::const_region                 const_region;
        typedef typename T_container::const_container           const_container;
    };
    
    // Iterator --------------------------------------------------------------//
    template <typename T_container>
    class base_iterator {    
        // Base class for 2D iterators
        public:     
            typedef std::bidirectional_iterator_tag                             iterator_category;   
            typedef typename T_container::value_type                                   value_type;
            typedef typename T_container::size_type                                     size_type; 
            typedef typename T_container::difference_type                         difference_type;         
            typedef typename T_container::coords                                           coords;    
            typedef typename container_traits<T_container>::pointer                       pointer; // Can be const or non-const
            typedef typename container_traits<T_container>::reference                   reference; // Can be const or non-const
            typedef typename container_traits<T_container>::nonconst_container nonconst_container;
            typedef typename container_traits<T_container>::container                   container; // Can be const or non-const
            typedef typename container_traits<T_container>::const_container       const_container;
            
            template <typename T_container2>
            friend class base_iterator;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//        
            base_iterator() noexcept : A_ptr(nullptr), p() { }
            base_iterator(const base_iterator&) noexcept = default;
            base_iterator(base_iterator&&) noexcept = default;
            base_iterator& operator=(const base_iterator&) = default;
            base_iterator& operator=(base_iterator&&) = default; 
            virtual ~base_iterator() noexcept = default;
                    
            // Additional Constructors ---------------------------------------//
            base_iterator(container &A, difference_type p) : A_ptr(&A), p(p) { }
            // Allow conversions from non const to const
            template<typename T_container2> 
            base_iterator(const base_iterator<T_container2> &it, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : A_ptr(it.A_ptr), p(it.p) { }  
            
            // Access methods ------------------------------------------------//
            difference_type pos() const { return p; }
            coords pos_2D() const { return A_ptr->ind2sub(p); }
            reference operator*() const;

            // Arithmetic methods --------------------------------------------//
            virtual base_iterator& operator++() = 0;
            virtual base_iterator& operator--() = 0;
            
            template <typename T_container2>
            typename std::enable_if<std::is_same<typename container_traits<T_container2>::nonconst_container, nonconst_container>::value, bool>::type operator==(const base_iterator<T_container2> &it) const { 
                return A_ptr == it.A_ptr && p == it.p; // enable if containers are the same
            }
            template <typename T_container2>
            typename std::enable_if<std::is_same<typename container_traits<T_container2>::nonconst_container, nonconst_container>::value, bool>::type operator!=(const base_iterator<T_container2> &it) const { 
                return !((*this) == it); // enable if containers are the same
            }
            
            // Clone ---------------------------------------------------------//
            virtual base_iterator* clone() const = 0; 
            virtual base_iterator<const_container>* const_clone() const = 0; 
            
        protected:                  
            // Utility -------------------------------------------------------//
            void chk_valid_increment() const; 
            void chk_valid_decrement() const;
            void chk_in_range() const; 
          
            container *A_ptr;
            difference_type p;     
    };  
        
    template <typename T_container> 
    class simple_iterator final : public base_iterator<T_container> { 
        // Simplest iterator
        public:     
            typedef typename base_iterator<T_container>::iterator_category   iterator_category;   
            typedef typename base_iterator<T_container>::value_type                 value_type;
            typedef typename base_iterator<T_container>::size_type                   size_type; 
            typedef typename base_iterator<T_container>::difference_type       difference_type; 
            typedef typename base_iterator<T_container>::coords                         coords;            
            typedef typename base_iterator<T_container>::pointer                       pointer; 
            typedef typename base_iterator<T_container>::reference                   reference; 
            typedef typename base_iterator<T_container>::nonconst_container nonconst_container;
            typedef typename base_iterator<T_container>::container                   container;
            typedef typename base_iterator<T_container>::const_container       const_container;
                        
            template <typename T_container2>
            friend class simple_iterator;
            
            // Rule of 5 and destructor --------------------------------------//        
            simple_iterator() noexcept = default;
            simple_iterator(const simple_iterator&) noexcept = default;
            simple_iterator(simple_iterator&&) noexcept = default;
            simple_iterator& operator=(const simple_iterator&) = default;
            simple_iterator& operator=(simple_iterator&&) = default; 
            ~simple_iterator() noexcept override = default;
            
            // Additional Constructors ---------------------------------------//
            simple_iterator(container &A, difference_type p) : base_iterator<container>(A,p) { } 
            // Allow conversions from non const to const
            template<typename T_container2> 
            simple_iterator(const simple_iterator<T_container2> &it, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_iterator<container>(it) { }  
            
            // Arithmetic methods --------------------------------------------//
            simple_iterator& operator++() override;
            simple_iterator& operator--() override;
            
            // Clone ---------------------------------------------------------//
            simple_iterator* clone() const override { return new simple_iterator(*this); } 
            simple_iterator<const_container>* const_clone() const override { return new simple_iterator<const_container>(*this); } 
    };
        
    template <typename T_container> 
    class sub_iterator final : public base_iterator<T_container> { 
        // Iterator for subarray
        public:     
            typedef typename base_iterator<T_container>::iterator_category   iterator_category;   
            typedef typename base_iterator<T_container>::value_type                 value_type;
            typedef typename base_iterator<T_container>::size_type                   size_type; 
            typedef typename base_iterator<T_container>::difference_type       difference_type; 
            typedef typename base_iterator<T_container>::coords                         coords;            
            typedef typename base_iterator<T_container>::pointer                       pointer; 
            typedef typename base_iterator<T_container>::reference                   reference; 
            typedef typename base_iterator<T_container>::nonconst_container nonconst_container;
            typedef typename base_iterator<T_container>::container                   container;
            typedef typename base_iterator<T_container>::const_container       const_container;
                        
            template <typename T_container2>
            friend class sub_iterator;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//        
            sub_iterator() noexcept : sub_p(), sub_h(), sub_w(), sub_s() { }
            sub_iterator(const sub_iterator&) noexcept = default;
            sub_iterator(sub_iterator&&) noexcept = default;
            sub_iterator& operator=(const sub_iterator&) = default;
            sub_iterator& operator=(sub_iterator&&) = default; 
            ~sub_iterator() noexcept override = default;
            
            // Additional Constructors ---------------------------------------//
            sub_iterator(container&, const coords&, const coords&, const coords&);
            // Allow conversions from non const to const
            template <typename T_container2>
            sub_iterator(const sub_iterator<T_container2> &it, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_iterator<container>(it), r_sub1_2D(it.r_sub1_2D), r_sub2_2D(it.r_sub2_2D), sub_p(it.sub_p), sub_h(it.sub_h), sub_w(it.sub_w), sub_s(it.sub_s) { }  
            
            // Arithmetic methods --------------------------------------------//
            sub_iterator& operator++() override;
            sub_iterator& operator--() override;
            
            // Clone ---------------------------------------------------------//
            sub_iterator* clone() const override { return new sub_iterator(*this); } 
            sub_iterator<const_container>* const_clone() const override { return new sub_iterator<const_container>(*this); } 
                    
        private:              
            void chk_valid_ranges() const;
            
            coords r_sub1_2D;  
            coords r_sub2_2D;    
            difference_type sub_p;  
            difference_type sub_h;
            difference_type sub_w; 
            difference_type sub_s;    
    };        
    
    template <typename T_container> 
    class bool_iterator final : public base_iterator<T_container> { 
        // Iterator for boolean array
        public:     
            typedef typename base_iterator<T_container>::iterator_category   iterator_category;   
            typedef typename base_iterator<T_container>::value_type                 value_type;
            typedef typename base_iterator<T_container>::size_type                   size_type; 
            typedef typename base_iterator<T_container>::difference_type       difference_type; 
            typedef typename base_iterator<T_container>::coords                         coords;            
            typedef typename base_iterator<T_container>::pointer                       pointer; 
            typedef typename base_iterator<T_container>::reference                   reference;
            typedef typename base_iterator<T_container>::nonconst_container nonconst_container; 
            typedef typename base_iterator<T_container>::container                   container;
            typedef typename base_iterator<T_container>::const_container       const_container;
            typedef typename T_container::bool_container                        bool_container;
                      
            template <typename T_container2>
            friend class bool_iterator;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//        
            bool_iterator() noexcept : A_bool_ptr(nullptr) { }
            bool_iterator(const bool_iterator&) noexcept = default;
            bool_iterator(bool_iterator&&) noexcept = default;
            bool_iterator& operator=(const bool_iterator&) = default;
            bool_iterator& operator=(bool_iterator&&) = default; 
            ~bool_iterator() noexcept override = default;            
            
            // Additional Constructors ---------------------------------------//
            bool_iterator(container&, difference_type, const bool_container*);  
            // Allow conversions from non const to const
            template <typename T_container2>
            bool_iterator(const bool_iterator<T_container2> &it, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_iterator<container>(it), A_bool_ptr(it.A_bool_ptr) { }  
            
            // Arithmetic methods --------------------------------------------//
            bool_iterator& operator++() override;
            bool_iterator& operator--() override;            
            
            // Clone ---------------------------------------------------------//
            bool_iterator* clone() const override { return new bool_iterator(*this); } 
            bool_iterator<const_container>* const_clone() const override { return new bool_iterator<const_container>(*this); } 
        
        private:                       
            // Utility -------------------------------------------------------//
            void chk_same_size() const;
            
            const bool_container *A_bool_ptr;
    };
    
    template <typename T_iterator> 
    class interface_iterator final {  
        // This is a container for the base iterator class to hide the 
        // inheritance tree.
        public:     
            typedef typename T_iterator::iterator_category    iterator_category;
            typedef typename T_iterator::value_type                  value_type;
            typedef typename T_iterator::size_type                    size_type; 
            typedef typename T_iterator::difference_type        difference_type; 
            typedef typename T_iterator::coords                          coords;
            typedef typename T_iterator::pointer                        pointer; 
            typedef typename T_iterator::reference                    reference; 
            typedef typename T_iterator::nonconst_container  nonconst_container;
            typedef typename T_iterator::container                    container;
            typedef typename T_iterator::const_container        const_container;
                                    
            template <typename T_iterator2>
            friend class interface_iterator;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//        
            interface_iterator() noexcept = default;
            interface_iterator(const interface_iterator &it) : ptr(it.ptr ? it.ptr->clone() : nullptr) { }
            interface_iterator(interface_iterator &&it) : ptr(std::move(it.ptr)) { }
            interface_iterator& operator=(const interface_iterator &it) { this->ptr.reset(it.ptr ? it.ptr->clone() : nullptr); return *this; }
            interface_iterator& operator=(interface_iterator &&it) { this->ptr = std::move(it.ptr); return *this; }
            ~interface_iterator() noexcept = default;
            
            // Additional Constructors ---------------------------------------//
            // explicit is important since ptr is wrapped in shared_ptr, which 
            // will call delete on ptr when this object goes out of scope.
            explicit interface_iterator(T_iterator *ptr) : ptr(ptr) { }
            // Allow conversions from non const to const
            template <typename T_iterator2>
            interface_iterator(const interface_iterator<T_iterator2> &it, typename std::enable_if<std::is_same<typename T_iterator2::container, nonconst_container>::value, int>::type = 0) : 
                ptr(it.ptr ? it.ptr->const_clone() : nullptr) { }
                    
            // Access methods ------------------------------------------------//            
            difference_type pos() const { return ptr->pos(); }
            coords pos_2D() const { return ptr->pos_2D(); }
            reference operator*() const { return *(*ptr); }

            // Arithmetic methods --------------------------------------------//
            interface_iterator& operator++() { ++(*ptr); return *this; }
            interface_iterator& operator--() { --(*ptr); return *this; }
            template <typename T_iterator2>
            typename std::enable_if<std::is_same<typename T_iterator2::nonconst_container, nonconst_container>::value, bool>::type operator==(const interface_iterator<T_iterator2> &it) const { 
                return *(it.ptr) == *(ptr); // enable if containers are the same
            }
            template <typename T_iterator2>
            typename std::enable_if<std::is_same<typename T_iterator2::nonconst_container, nonconst_container>::value, bool>::type operator!=(const interface_iterator<T_iterator2> &it) const { 
                return *(it.ptr) != *(ptr); // enable if containers are the same
            }

        private:        
            std::shared_ptr<T_iterator> ptr;            
    };    
    
    // Region ----------------------------------------------------------------//
    template <typename T_container> 
    class base_region {    
        // Base class for regions.
        public:      
            typedef typename T_container::value_type                                   value_type;
            typedef typename T_container::size_type                                     size_type; 
            typedef typename T_container::difference_type                         difference_type;   
            typedef typename T_container::coords                                           coords;
            typedef typename container_traits<T_container>::pointer                       pointer; // Can be const or non-const
            typedef typename container_traits<T_container>::const_pointer           const_pointer; 
            typedef typename container_traits<T_container>::reference                   reference; // Can be const or non-const
            typedef typename container_traits<T_container>::const_reference       const_reference; 
            typedef typename container_traits<T_container>::iterator                     iterator; // Can be const or non-const
            typedef typename container_traits<T_container>::const_iterator         const_iterator; 
            typedef typename container_traits<T_container>::nonconst_container nonconst_container;
            typedef typename container_traits<T_container>::container                   container; // Can be const or non-const
            typedef typename container_traits<T_container>::const_container       const_container;
            
            template <typename T_container2>
            friend class base_region;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            base_region() noexcept : A_ptr(nullptr), region_h(), region_w(), region_s() { }
            base_region(const base_region&) noexcept = default;
            base_region(base_region&&) noexcept = default;
            base_region& operator=(const base_region&);  
            base_region& operator=(base_region &&reg) { return operator=(reg); }
            virtual ~base_region() noexcept = default;
            
            // Additional Constructors ---------------------------------------//
            base_region(container &A, difference_type region_h = 0, difference_type region_w = 0) : A_ptr(&A), region_h(region_h), region_w(region_w), region_s(region_h*region_w) { }
            // Allow conversions from non const to const
            template<typename T_container2> 
            base_region(const base_region<T_container2> &reg, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) :
                A_ptr(reg.A_ptr), region_h(reg.region_h), region_w(reg.region_w), region_s(reg.region_s) { }  
            
            // Assignment methods --------------------------------------------//
            // Allow assignments from const_regions to region. Since this 
            // conversion doesnt exist, make assignment operator a template 
            // function
            template<typename T_container2> 
            typename std::enable_if<std::is_same<typename container_traits<T_container2>::nonconst_container, nonconst_container>::value, base_region&>::type operator=(const base_region<T_container2>&);  
            base_region& operator=(const_container&);  
            base_region& operator=(const_reference);
            
            // Access methods ------------------------------------------------//
            difference_type region_height() const { return region_h; }
            difference_type region_width() const { return region_w; }
            difference_type region_size() const { return region_s; }
                        
            // Clone ---------------------------------------------------------//
            virtual base_region* clone() const = 0; 
            virtual base_region<const_container>* const_clone() const = 0; 
            
            // Iterators -----------------------------------------------------//
            // Note that iterators have a dependence on both the container and
            // region they are called from. If region OR Array2D is destroyed,
            // iterators may become invalid.
            virtual iterator begin() const = 0;
            virtual iterator end() const = 0;
            virtual const_iterator cbegin() const = 0;
            virtual const_iterator cend() const = 0;
            
            // Utility -------------------------------------------------------//
            std::string region_size_string() const { return std::to_string(region_s); }   
            std::string region_size_2D_string() const { return "(" + std::to_string(region_h) + "," + std::to_string(region_w) + ")"; }
        
        protected:           
            container *A_ptr;
            difference_type region_h;
            difference_type region_w;
            difference_type region_s;
    };    
            
    template <typename T_container> 
    class simple_region final : public base_region<T_container> { 
        // Simplest region
        public:       
            typedef typename base_region<T_container>::value_type                 value_type;
            typedef typename base_region<T_container>::size_type                   size_type; 
            typedef typename base_region<T_container>::difference_type       difference_type;  
            typedef typename base_region<T_container>::coords                         coords;
            typedef typename base_region<T_container>::pointer                       pointer; 
            typedef typename base_region<T_container>::const_pointer           const_pointer; 
            typedef typename base_region<T_container>::reference                   reference; 
            typedef typename base_region<T_container>::const_reference       const_reference;    
            typedef typename base_region<T_container>::iterator                     iterator;  
            typedef typename base_region<T_container>::const_iterator         const_iterator;   
            typedef typename base_region<T_container>::nonconst_container nonconst_container; 
            typedef typename base_region<T_container>::container                   container;
            typedef typename base_region<T_container>::const_container       const_container;
            
            template <typename T_container2>
            friend class simple_region;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            simple_region() = default;
            simple_region(const simple_region&) noexcept = default;
            simple_region(simple_region&&) noexcept = default; 
            simple_region& operator=(const simple_region &reg) { base_region<container>::operator=(reg); return *this; } 
            simple_region& operator=(simple_region &&reg) { return operator=(reg); }
            ~simple_region() noexcept override = default; 
                        
            // Additional Constructors ---------------------------------------//
            simple_region(container&, const coords&); 
            // Allow conversions from non const to const
            template<typename T_container2> 
            simple_region(const simple_region<T_container2> &reg, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_region<container>(reg), r(reg.r) { }  
            
            // Clone ---------------------------------------------------------//
            simple_region* clone() const override { return new simple_region(*this); }    
            simple_region<const_container>* const_clone() const override { return new simple_region<const_container>(*this); }                       
            
            // Iterators -----------------------------------------------------//
            iterator begin() const override { return iterator(new simple_iterator<container>(*this->A_ptr,r.first)); }
            iterator end() const override { return iterator(new simple_iterator<container>(*this->A_ptr,r.second)); }
            const_iterator cbegin() const override { return const_iterator(new simple_iterator<const_container>(*this->A_ptr,r.first)); }
            const_iterator cend() const override { return const_iterator(new simple_iterator<const_container>(*this->A_ptr,r.second)); }
            
        private:                        
            // Utility -------------------------------------------------------//
            void chk_valid_range() const;
            
            coords r;
    }; 
                
    template <typename T_container> 
    class sub_region final : public base_region<T_container> { 
        // Region for subarray
        public:   
            typedef typename base_region<T_container>::value_type                 value_type;
            typedef typename base_region<T_container>::size_type                   size_type; 
            typedef typename base_region<T_container>::difference_type       difference_type;  
            typedef typename base_region<T_container>::coords                         coords;
            typedef typename base_region<T_container>::pointer                       pointer; 
            typedef typename base_region<T_container>::const_pointer           const_pointer; 
            typedef typename base_region<T_container>::reference                   reference; 
            typedef typename base_region<T_container>::const_reference       const_reference;    
            typedef typename base_region<T_container>::iterator                     iterator;  
            typedef typename base_region<T_container>::const_iterator         const_iterator;   
            typedef typename base_region<T_container>::nonconst_container nonconst_container; 
            typedef typename base_region<T_container>::container                   container;
            typedef typename base_region<T_container>::const_container       const_container;
            
            template <typename T_container2>
            friend class sub_region;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//  
            sub_region() noexcept : sub_h(), sub_w(), sub_s() { }
            sub_region(const sub_region&) noexcept = default;
            sub_region(sub_region&&) noexcept = default; 
            sub_region& operator=(const sub_region &reg) { base_region<container>::operator=(reg); return *this; } 
            sub_region& operator=(sub_region &&reg) { return operator=(reg); }
            ~sub_region() noexcept override = default;           
            
            // Additional Constructors ---------------------------------------//
            sub_region(container&, const coords&, const coords&);
            // Allow conversions from non const to const
            template<typename T_container2> 
            sub_region(const sub_region<T_container2> &reg, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_region<container>(reg), r_sub1_2D(reg.r_sub1_2D), r_sub2_2D(reg.r_sub2_2D), sub_h(reg.sub_h), sub_w(reg.sub_w), sub_s(reg.sub_s) { }  
                        
            // Clone ---------------------------------------------------------//
            sub_region* clone() const override { return new sub_region(*this); } 
            sub_region<const_container>* const_clone() const override { return new sub_region<const_container>(*this); }    
                    
            // Iterators -----------------------------------------------------//
            // Check sub_s equals to zero for empty sub regions because sub height 
            // can be zero with a nonzero sub width and vice versa.
            iterator begin() const override { return iterator(new sub_iterator<container>(*this->A_ptr,{r_sub1_2D.first, r_sub2_2D.first},r_sub1_2D,r_sub2_2D)); }
            iterator end() const override { return sub_s == 0 ? begin() : iterator(new sub_iterator<container>(*this->A_ptr,{r_sub1_2D.first, r_sub2_2D.second},r_sub1_2D,r_sub2_2D)); }
            const_iterator cbegin() const override { return const_iterator(new sub_iterator<const_container>(*this->A_ptr,{r_sub1_2D.first, r_sub2_2D.first},r_sub1_2D,r_sub2_2D)); }
            const_iterator cend() const override { return sub_s == 0 ? cbegin() : const_iterator(new sub_iterator<const_container>(*this->A_ptr,{r_sub1_2D.first, r_sub2_2D.second},r_sub1_2D,r_sub2_2D)); }
            
        private:
            // Utility -------------------------------------------------------//
            void chk_valid_ranges() const;
            
            coords r_sub1_2D;  
            coords r_sub2_2D;    
            difference_type sub_h;
            difference_type sub_w;
            difference_type sub_s;
    };
            
    template <typename T_container> 
    class bool_region final : public base_region<T_container> { 
        // Region for logical indexing 
        public:          
            typedef typename base_region<T_container>::value_type                 value_type;
            typedef typename base_region<T_container>::size_type                   size_type; 
            typedef typename base_region<T_container>::difference_type       difference_type;  
            typedef typename base_region<T_container>::coords                         coords;
            typedef typename base_region<T_container>::pointer                       pointer; 
            typedef typename base_region<T_container>::const_pointer           const_pointer; 
            typedef typename base_region<T_container>::reference                   reference; 
            typedef typename base_region<T_container>::const_reference       const_reference;    
            typedef typename base_region<T_container>::iterator                     iterator;  
            typedef typename base_region<T_container>::const_iterator         const_iterator;   
            typedef typename base_region<T_container>::nonconst_container nonconst_container; 
            typedef typename base_region<T_container>::container                   container;
            typedef typename base_region<T_container>::const_container       const_container;
            typedef typename T_container::bool_container                      bool_container;
            
            template <typename T_container2>
            friend class bool_region;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            bool_region() noexcept = default;
            bool_region(const bool_region&) = default;
            bool_region(bool_region&&) noexcept = default;
            bool_region& operator=(const bool_region &reg) { base_region<container>::operator=(reg); return *this; } 
            bool_region& operator=(bool_region &&reg) { return operator=(reg); } 
            ~bool_region() noexcept override = default;           
            
            // Additional Constructors ---------------------------------------//
            bool_region(container&, bool_container); // by-value
            // Allow conversions from non const to const
            template<typename T_container2> 
            bool_region(const bool_region<T_container2> &reg, typename std::enable_if<std::is_same<T_container2, nonconst_container>::value, int>::type = 0) : 
                base_region<container>(reg), A_bool_ptr(reg.A_bool_ptr) { }  
            
            // Clone ---------------------------------------------------------//
            bool_region* clone() const override { return new bool_region(*this); } 
            bool_region<const_container>* const_clone() const override { return new bool_region<const_container>(*this); }
                    
            // Iterators -----------------------------------------------------//
            iterator begin() const override { return iterator(new bool_iterator<container>(*this->A_ptr,0,A_bool_ptr.get())); }
            iterator end() const override { return iterator(new bool_iterator<container>(*this->A_ptr,this->A_ptr->size(),A_bool_ptr.get())); }
            const_iterator cbegin() const override { return const_iterator(new bool_iterator<const_container>(*this->A_ptr,0,A_bool_ptr.get())); }
            const_iterator cend() const override { return const_iterator(new bool_iterator<const_container>(*this->A_ptr,this->A_ptr->size(),A_bool_ptr.get())); }
            
        private: 
            // Utility -------------------------------------------------------//
            void chk_same_size() const;
                       
            std::shared_ptr<bool_container> A_bool_ptr; 
    };
    
    template <typename T_region> 
    class interface_region final { 
        // This is a container for the base region class to hide the inheritance 
        // tree.
        public:     
            typedef typename T_region::value_type                    value_type;
            typedef typename T_region::size_type                      size_type; 
            typedef typename T_region::difference_type          difference_type;      
            typedef typename T_region::coords                            coords;  
            typedef typename T_region::pointer                          pointer; 
            typedef typename T_region::const_pointer              const_pointer; 
            typedef typename T_region::reference                      reference;  
            typedef typename T_region::const_reference          const_reference;     
            typedef typename T_region::iterator                        iterator; 
            typedef typename T_region::const_iterator            const_iterator;   
            typedef typename T_region::nonconst_container    nonconst_container;
            typedef typename T_region::container                      container;
            typedef typename T_region::const_container          const_container;
             
            template <typename T_region2>
            friend class interface_region;
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            interface_region() noexcept = default;
            interface_region(const interface_region &reg) : ptr(reg.ptr ? reg.ptr->clone() : nullptr) { }
            interface_region(interface_region &&reg) : ptr(std::move(reg.ptr)) { }
            // Note that region is special in that the assignment operator 
            // is overwritten. Do not store regions as members of objects - as 
            // default assignment will not work properly.
            interface_region& operator=(const interface_region &reg) { (*this->ptr) = *(reg.ptr); return *this; }
            interface_region& operator=(interface_region &&reg) { return operator=(reg); }  
            ~interface_region() noexcept = default;
            
            // Additional Constructors ---------------------------------------//
            // explicit is important since ptr is wrapped in shared_ptr, which 
            // will call delete on ptr when this object goes out of scope.
            explicit interface_region(T_region *ptr) : ptr(ptr) { }
            // Allow conversions from non const to const           
            template <typename T_region2>
            interface_region(const interface_region<T_region2> &reg, typename std::enable_if<std::is_same<typename T_region2::container, nonconst_container>::value, int>::type = 0) : 
                ptr(reg.ptr ? reg.ptr->const_clone() : nullptr) { }
                                
            // Assignment methods --------------------------------------------//
            template <typename T_region2>
            typename std::enable_if<std::is_same<typename T_region2::nonconst_container, nonconst_container>::value, interface_region&>::type operator=(const T_region2 &reg) { *(ptr) = *(reg.ptr); return *this; }  
            interface_region& operator=(const_container &A) { *(ptr) = A; return *this; }  
            interface_region& operator=(const_reference val) { *(ptr) = val; return *this; }
            
            // Access methods ------------------------------------------------//            
            difference_type region_height() const { return ptr->region_height(); }
            difference_type region_width() const { return ptr->region_width(); }
            difference_type region_size() const { return ptr->region_size(); }
                        
            // Iterators -----------------------------------------------------//    
            iterator begin() const { return ptr->begin(); }
            iterator end() const { return ptr->end(); }
            const_iterator cbegin() const { return ptr->cbegin(); } 
            const_iterator cend() const { return ptr->cend(); }
            
            // Utility -------------------------------------------------------//
            std::string region_size_string() const { return ptr->region_size_string(); }   
            std::string region_size_2D_string() const { return ptr->region_size_2D_string(); }   
            
        private:        
            std::shared_ptr<T_region> ptr;            
    };  
    
    // Interpolator ----------------------------------------------------------//
    template <typename T_container> 
    class base_interp {    
        // Base class for interpolator. Note that this class does not take into
        // account if caller Array2D is const or not.
        public:      
            typedef typename T_container::value_type                             value_type;
            typedef typename T_container::reference                               reference;
            typedef typename T_container::size_type                               size_type; 
            typedef typename T_container::difference_type                   difference_type;         
            typedef typename T_container::coords                                     coords;        
            typedef typename container_traits<T_container>::nonconst_container    container; 
            typedef typename container_traits<T_container>::const_container const_container; 
            
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            base_interp() : A_ptr(nullptr) { }
            base_interp(const base_interp&) = default;
            base_interp(base_interp&&) noexcept = default;
            base_interp& operator=(const base_interp&) = default;  
            base_interp& operator=(base_interp&&) = default;
            virtual ~base_interp() noexcept = default;
            
            // Additional Constructors ---------------------------------------//            
            base_interp(const_container &A) : A_ptr(&A), first_order_buf(3,1) { }
            
            // Arithmetic methods --------------------------------------------//
            virtual value_type operator()(double, double) const = 0;
            // first_order returns interpolation value along with it's first
            // order gradients. Interpolation value is provided with them 
            // because often times calculations are reused when calculating 
            // gradients along with the interpolation value, and also all three 
            // values are usually needed together anyway.
            virtual const_container& first_order(double, double) const = 0; 
                        
            // Clone ---------------------------------------------------------//
            virtual base_interp* clone() const = 0; 
            
        protected:
            // Utility -------------------------------------------------------//
            virtual bool out_of_bounds(double, double) const = 0; 
            
            const_container *A_ptr;
            mutable container first_order_buf;
    };   
    
    template <typename T_container> 
    class nearest_interp final : public base_interp<T_container> { 
        // Nearest neighbor interpolation
        public:          
            typedef typename base_interp<T_container>::value_type           value_type;
            typedef typename base_interp<T_container>::reference             reference;  
            typedef typename base_interp<T_container>::size_type             size_type; 
            typedef typename base_interp<T_container>::difference_type difference_type;  
            typedef typename base_interp<T_container>::coords                   coords; 
            typedef typename base_interp<T_container>::container             container;
            typedef typename base_interp<T_container>::const_container const_container;
            
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            nearest_interp() = default;
            nearest_interp(const nearest_interp&) = default;
            nearest_interp(nearest_interp&&) noexcept = default;
            nearest_interp& operator=(const nearest_interp&) = default;  
            nearest_interp& operator=(nearest_interp&&) = default;
            ~nearest_interp() noexcept = default;
            
            // Additional Constructors ---------------------------------------//            
            nearest_interp(const_container &A) : base_interp<container>(A) { }
            
            // Arithmetic methods --------------------------------------------//
            value_type operator()(double, double) const override;
            const_container& first_order(double, double) const override;
            
            // Clone ---------------------------------------------------------//
            nearest_interp* clone() const override { return new nearest_interp(*this); }
            
        protected:
            // Utility -------------------------------------------------------//
            // Since nearest neighbor rounds, the bounds can be extended by 0.5
            bool out_of_bounds(double p1, double p2) const override { return p1 <= -0.5 || p2 <= -0.5 || p1 >= this->A_ptr->height() - 0.5 || p2 >= this->A_ptr->width() - 0.5; }
    }; 
    
    template <typename T_container> 
    class linear_interp final : public base_interp<T_container> {
        // Bilinear interpolation - not implemented as coefficient matrix 
        // interpolation since it's simple and fast.
        public:          
            typedef typename base_interp<T_container>::value_type           value_type;
            typedef typename base_interp<T_container>::reference             reference;  
            typedef typename base_interp<T_container>::size_type             size_type; 
            typedef typename base_interp<T_container>::difference_type difference_type;  
            typedef typename base_interp<T_container>::coords                   coords; 
            typedef typename base_interp<T_container>::container             container;
            typedef typename base_interp<T_container>::const_container const_container;
            
            friend container;
            
            // Rule of 5 and destructor --------------------------------------//
            linear_interp() = default;
            linear_interp(const linear_interp&) = default;
            linear_interp(linear_interp&&) noexcept = default;
            linear_interp& operator=(const linear_interp&) = default;  
            linear_interp& operator=(linear_interp&&) = default;
            ~linear_interp() noexcept = default;
                                                
            // Additional Constructors ---------------------------------------//            
            linear_interp(const_container &A) : base_interp<container>(A) { }
            
            // Arithmetic methods --------------------------------------------//
            value_type operator()(double, double) const override;
            const_container& first_order(double, double) const override;
            
            // Clone ---------------------------------------------------------//
            linear_interp* clone() const override { return new linear_interp(*this); }

        private:            
            // Utility -------------------------------------------------------//
            // Uses four points around floored point. Possibly add scheme to 
            // interpolation points on the right and bottom edges of the image 
            // later.
            bool out_of_bounds(double p1, double p2) const override { return p1 < 0 || p1 >= this->A_ptr->height() - 1 || p2 < 0 || p2 >= this->A_ptr->width() - 1; }
    }; 
    
    template <typename T_container> 
    class coef_mat_interp_base : public base_interp<T_container> {
        // The is the base class for interpolation schemes which use a coefficient
        // matrix
        public:          
            typedef typename base_interp<T_container>::value_type           value_type;
            typedef typename base_interp<T_container>::reference             reference;  
            typedef typename base_interp<T_container>::size_type             size_type; 
            typedef typename base_interp<T_container>::difference_type difference_type;  
            typedef typename base_interp<T_container>::coords                   coords;
            typedef typename base_interp<T_container>::container             container;
            typedef typename base_interp<T_container>::const_container const_container;
            
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
                base_interp<container>(A), order(order), p1_pow_buf(order+1,1), p2_pow_buf(order+1,1), p1_pow_dp1_buf(order+1,1), p2_pow_dp2_buf(order+1,1) { 
                if (order < 1) {
                    // This is purely a programmer error since this class is abstract
                    throw std::invalid_argument("Attempted to form coef_mat_interp_base with order of: " + std::to_string(order) + " order must be 1 or greater.");
                }
            }
                        
            // Arithmetic methods --------------------------------------------//
            value_type operator()(double, double) const override;
            const_container& first_order(double, double) const override;
            
        protected:                 
            // Arithmetic methods --------------------------------------------//
            virtual container& get_p_pow(container&, double) const;
            virtual container& get_dp_pow(container&, const_container&) const;
            virtual value_type t_vec_mat_vec(const_container&, const_container&, const_container&) const;
            virtual const_container& calc_coef_mat(container&, const_container&, difference_type, difference_type) const = 0;
            virtual const_container& get_coef_mat(difference_type, difference_type) const = 0;
            
            difference_type order;
            mutable container p1_pow_buf;
            mutable container p2_pow_buf;
            mutable container p1_pow_dp1_buf;
            mutable container p2_pow_dp2_buf;
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
            
        protected:            
            // Arithmetic methods --------------------------------------------//
            const_container& calc_coef_mat(container&, const_container&, difference_type, difference_type) const override;
            
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
            cubic_interp(const_container &A) : cubic_interp_base<container>(A), coef_mat_buf(4,4) { }
            
            // Clone ---------------------------------------------------------//
            cubic_interp* clone() const override { return new cubic_interp(*this); }
            
        private:
            // Arithmetic methods --------------------------------------------//
            // This will compute the coefficient matrix
            const_container& get_coef_mat(difference_type p1, difference_type p2) const override {
                return this->calc_coef_mat(coef_mat_buf, *this->A_ptr, p1 - 1, p2 - 1);
            }
            
            mutable container coef_mat_buf;  
    }; 

    template <typename T_container> 
    class cubic_interp_precompute final : public cubic_interp_base<T_container> { 
        // Bicubic interpolation with a precomputed coefficient table
        // Note that this requires a LOT of memory (approx 16 x the original image), so 
        // only use this if the same image is interpolated many times.
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
            cubic_interp_precompute() = default;
            cubic_interp_precompute(const cubic_interp_precompute&) = default;
            cubic_interp_precompute(cubic_interp_precompute&&) noexcept = default;
            cubic_interp_precompute& operator=(const cubic_interp_precompute&) = default;  
            cubic_interp_precompute& operator=(cubic_interp_precompute&&) = default;
            ~cubic_interp_precompute() noexcept = default;
                        
            // Additional Constructors ---------------------------------------//            
            cubic_interp_precompute(const_container&);
            
            // Clone ---------------------------------------------------------//
            cubic_interp_precompute* clone() const override { return new cubic_interp_precompute(*this); }
            
        private:
            // Arithmetic methods --------------------------------------------//
            // This returns precomputed coef matrix instead of calculating it
            const_container& get_coef_mat(difference_type p1, difference_type p2) const override { return (*coef_mat_precompute_ptr)(p1,p2); }
                        
            std::shared_ptr<Array2D<container>> coef_mat_precompute_ptr; // immutable
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
            quintic_interp_base(const_container &A) : coef_mat_interp_base<container>(A,5) { }
                                              
            // Arithmetic methods --------------------------------------------//
            value_type operator()(double, double) const override;
            const_container& first_order(double, double) const override;
            
        protected:            
            // Arithmetic methods --------------------------------------------//
            std::shared_ptr<container> get_bspline_mat_ptr(const_container&) const;
            const_container& calc_coef_mat(container&, const_container&, difference_type, difference_type) const override;
            
            // Utility -------------------------------------------------------//
            // Uses 36 points around floored point. Allow interpolation within 
            // entire image bounds since padding is used for the b-coefficient 
            // array; this padding must be 3 or greater.
            bool out_of_bounds(double p1, double p2) const override { return p1 < 0 || p1 > this->A_ptr->height() - 1 || p2 < 0 || p2 > this->A_ptr->width() - 1; }
            
            // Must be greater than or equal to 3 in order to interpolate entire 
            // image for any input size. Large borders mitigate ringing errors.
            difference_type bcoef_border = 20; 
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
            
        private:
            // Arithmetic methods --------------------------------------------//
            // The will compute the coefficient matrix
            const_container& get_coef_mat(difference_type p1, difference_type p2) const override {
                return this->calc_coef_mat(coef_mat_buf, *bcoef_ptr, p1 + this->bcoef_border - 2, p2 + this->bcoef_border - 2);
            }
                        
            mutable container coef_mat_buf;   
            std::shared_ptr<container> bcoef_ptr; // immutable
    }; 

    template <typename T_container> 
    class quintic_interp_precompute final : public quintic_interp_base<T_container> { 
        // Biquintic B-spline interpolation with a precomputed coefficient table
        // Note that this requires a LOT of memory (36 x the original image), so 
        // only use this if the same image is interpolated many times.
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
            quintic_interp_precompute() = default;
            quintic_interp_precompute(const quintic_interp_precompute&) = default;
            quintic_interp_precompute(quintic_interp_precompute&&) noexcept = default;
            quintic_interp_precompute& operator=(const quintic_interp_precompute&) = default;  
            quintic_interp_precompute& operator=(quintic_interp_precompute&&) = default;
            ~quintic_interp_precompute() noexcept = default;
                                    
            // Additional Constructors ---------------------------------------//            
            quintic_interp_precompute(const_container&);
            
            // Clone ---------------------------------------------------------//
            quintic_interp_precompute* clone() const override { return new quintic_interp_precompute(*this); }
            
        private:
            // Arithmetic methods --------------------------------------------//
            // This returns precomputed coef matrix instead of calculating it
            const_container& get_coef_mat(difference_type p1, difference_type p2) const override { return (*coef_mat_precompute_ptr)(p1,p2); }
            
            std::shared_ptr<Array2D<container>> coef_mat_precompute_ptr; // immutable
    }; 

    template <typename T_interp> 
    class interface_interp final { 
        // This is a container for the base interp class to hide the inheritance 
        // tree. 
        public:     
            typedef typename T_interp::value_type                    value_type;
            typedef typename T_interp::reference                      reference;    
            typedef typename T_interp::size_type                      size_type; 
            typedef typename T_interp::difference_type          difference_type;      
            typedef typename T_interp::coords                            coords;    
            typedef typename T_interp::container                      container;
            typedef typename T_interp::const_container          const_container;
             
            template <typename T_interp2>
            friend class interface_interp;
            friend container;
            
            // Part of rule of 5 ---------------------------------------------//
            interface_interp() noexcept : ptr(nullptr) { }
            interface_interp(const interface_interp &interp) : ptr(interp.ptr ? interp.ptr->clone() : nullptr) { }
            interface_interp(interface_interp &&interp) : ptr(std::move(interp.ptr)) { }
            interface_interp& operator=(const interface_interp &interp) { ptr.reset(interp.ptr ? interp.ptr->clone() : nullptr); return *this; }
            interface_interp& operator=(interface_interp &&interp) { ptr = std::move(interp.ptr); return *this; }  
            ~interface_interp() noexcept = default;
                                
            // Additional Constructors ---------------------------------------//
            // explicit is important since ptr is wrapped in shared_ptr, which 
            // will call delete on ptr when this object goes out of scope.
            explicit interface_interp(T_interp *ptr) : ptr(ptr) { }
            
            // Access methods ------------------------------------------------//
            value_type operator()(double p1, double p2) const { return (*ptr)(p1,p2); }
            const_container& first_order(double p1, double p2) const { return ptr->first_order(p1,p2); }
            
        private:        
            std::shared_ptr<T_interp> ptr;            
    }; 
    
}

// ---------------------------------------------------------------------------//
// Definitions ---------------------------------------------------------------//
// ---------------------------------------------------------------------------//

// Array2D -------------------------------------------------------------------//
template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator=(const Array2D &A) { 
    // Must copy before freeing in the case "this" is self-assigned
    auto new_ptr = allocate_and_copy(A.s,A.ptr);
    destroy_and_deallocate();
    // Copy size of A and assign new pointer
    this->h = A.h;
    this->w = A.w;
    this->s = A.s;
    this->ptr = new_ptr; 
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator=(Array2D &&A) noexcept {
    // Test to make sure this is not self-assigned
    if (this != &A) {
        destroy_and_deallocate();
        // Pointer is directly copied; make r-value A null so that its destructor 
        // does not free this pointer while also remaining valid
        this->h = A.h;
        this->w = A.w;
        this->s = A.s;
        this->ptr = A.ptr;
        A.make_null();
    }
    
    return *this;
}       

// Additional Constructors ---------------------------------------------------//
template <typename T, typename T_alloc>  
Array2D<T,T_alloc>::Array2D(difference_type h, difference_type w, const_reference val) : h(h), w(w), s(h*w), ptr(nullptr) { 
    chk_minsize_op(0,0,"Array2D construction"); 
    
    this->ptr = allocate_and_init(this->h*this->w, val);
} 

template <typename T, typename T_alloc>  
Array2D<T,T_alloc>::Array2D(std::initializer_list<value_type> il) : h(il.size()), w(1), s(h*w), ptr(allocate_and_init(s, value_type())) {
    // Regular initializer_list is initialized as a column; copy values.
    std::copy(il.begin(), il.end(), this->ptr);
}

template <typename T, typename T_alloc>  
Array2D<T,T_alloc>::Array2D(std::initializer_list<std::initializer_list<value_type>> il_row) : h(il_row.size()), w(il_row.begin()->size()), s(h*w), ptr(allocate_and_init(s, value_type())) {
    // Assign width based on first inner list; must check for consistency among 
    // every inner list and throw exception if initializer list is jagged.
    difference_type p1 = 0;
    for (auto it_row = il_row.begin(); it_row != il_row.end(); ++it_row) {
        // Get inner list
        auto il_col = *it_row;            
        // Must insure consistent width
        if (il_col.size() != this->w) {
            // This is a programmer error. Call destructor before throwing 
            // exception since memory was already allocated. Destructor call is
            // safe because all elements are initialized as default elements.
            this->~Array2D();
            
            throw std::invalid_argument("Initializer list for Array2D must not be jagged.");
        }

        difference_type p2 = 0;
        for (auto it_col = il_col.begin(); it_col != il_col.end(); ++it_col) {
            (*this)(p1,p2) = *it_col;                
            ++p2;
        }
        ++p1;
    }
}

// Static factory methods ----------------------------------------------------//
template <typename T, typename T_alloc>  
template <typename T_output>    
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T, T_alloc>::load(const std::string &filename) {    
    // Form stream
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading Array2D.");
    }
    
    // Form Array using stream static factory method
    auto A = Array2D::load(is);
    
    // Close stream
    is.close();
    
    return A;
}     

template <typename T, typename T_alloc>  
template <typename T_output>    
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T, T_alloc>::load(std::ifstream &is) { 
    // Form empty A and then fill in values in accordance to how they are saved
    Array2D A;
    
    // Load height -> width
    is.read(reinterpret_cast<char*>(&A.h), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&A.w), std::streamsize(sizeof(difference_type)));    
    A.s = A.h*A.w; // Set s
    
    // Allocate memory
    A.ptr = A.allocate_and_init(A.s, value_type());
    
    // Read data
    is.read(reinterpret_cast<char*>(A.ptr), std::streamsize(sizeof(value_type) * A.s));
    
    return A;
}  

// Conversion ----------------------------------------------------------------//
template <typename T, typename T_alloc> 
inline Array2D<T,T_alloc>::operator value_type() { 
    chk_size_op(1,1,"value_type conversion");
    
    return (*this)(0); 
}

// Indexing ------------------------------------------------------------------//
template <typename T, typename T_alloc>
inline const T& Array2D<T,T_alloc>::operator()(difference_type p) const {
    #ifndef NDEBUG
    // Checks make indexing slow, so only do so in debug mode
    chk_minsize_op(1,1,"1D single-element indexing");
    chk_in_bounds_op(p,"1D single-element indexing");
    #endif

    return ptr[p];
}      

template <typename T, typename T_alloc> 
inline const T& Array2D<T,T_alloc>::operator()(difference_type p1, difference_type p2) const {
    #ifndef NDEBUG
    // Checks make indexing slow, so only do so in debug mode
    chk_minsize_op(1,1,"2D single-element indexing");
    chk_in_bounds_op(p1,p2,"2D single-element indexing");
    #endif

    return ptr[sub2ind(p1,p2)];
}          

template <typename T, typename T_alloc> 
template <typename T_output>
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T,T_alloc>::this_stream(std::ostream &os) const {   
    // Note this prints in row-major order
    for (difference_type p1 = 0; p1 < h; ++p1) {
        for (difference_type p2 = 0; p2 < w; ++p2) {
            os << (*this)(p1,p2) << ((p2 < w - 1) ? '\t' : '\0');
        }
        os << '\n';
    }

    return os;
}

template <typename T, typename T_alloc> 
Array2D<T, T_alloc> Array2D<T, T_alloc>::this_repmat(difference_type rows, difference_type cols) const {        
    Array2D B(h * rows, w * cols);
    for (difference_type p2 = 0; p2 < cols; ++p2) {
        for (difference_type p1 = 0; p1 < rows; ++p1) {
            B({p1*h, (p1+1)*h-1},{p2*w, (p2+1)*w-1}) = (*this);
        }
    }

    return B;  
}        

template <typename T, typename T_alloc> 
Array2D<T, T_alloc> Array2D<T, T_alloc>::this_pad(difference_type padding, PAD type) const {        
    Array2D A(h + 2*padding, w + 2*padding);
    A({padding, h + padding - 1},{padding, w + padding - 1}) = *this; 
    switch (type) {
        case PAD::EXPAND_EDGES : 
            // fills in padding with edge values - check to make sure this array 
            // is not empty first
            if (!empty()) {
                // Fill corners:
                A({0,padding-1},{0,padding-1}) = (*this)(0,0);
                A({0,padding-1},{w+padding,w+2*padding-1}) = (*this)(0,last);
                A({h+padding,last},{0,padding-1}) = (*this)(last,0);
                A({h+padding,last},{w+padding,w+2*padding-1}) = (*this)(last,last);
                // Fill sides:
                A({0,padding-1},{padding,w+padding-1}) = repmat((*this)(0,all),padding,1);
                A({padding,h+padding-1},{0,padding-1}) = repmat((*this)(all,0),1,padding);
                A({padding,h+padding-1},{w+padding,last}) = repmat((*this)(all,last),1,padding);
                A({h+padding,last},{padding,w+padding-1}) = repmat((*this)(last,all),padding,1);
            }
            break;
        case PAD::ZEROS :
            break;
    }

    return A;  
}   

template <typename T, typename T_alloc> 
Array2D<T, T_alloc> Array2D<T, T_alloc>::this_t() const {        
    Array2D A(w, h);
    for (difference_type p2 = 0; p2 < w; ++p2) {
        for (difference_type p1 = 0; p1 < h; ++p1) {
            A(p2,p1) = (*this)(p1,p2);
        }
    }

    return A;  
}     

template <typename T, typename T_alloc>  
template <typename T_output>
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T, T_alloc>::this_save(const std::string &filename) const {     
    // Form stream
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving Array2D.");
    }
    
    // Save data into stream
    this_save(os);
        
    // Close stream
    os.close();
}      

template <typename T, typename T_alloc>  
template <typename T_output>
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T, T_alloc>::this_save(std::ofstream &os) const {        
    // Save height -> width -> data
    os.write(reinterpret_cast<const char*>(&h), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&w), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(ptr), std::streamsize(sizeof(value_type) * s));
}  


// Logical operations --------------------------------------------------------//
template <typename T, typename T_alloc> 
bool Array2D<T, T_alloc>::this_isequal(const Array2D &A) const { 
    if (h != A.h || w != A.w) {
        return false;
    }

    for (difference_type p = 0; p < s; ++p) { 
        if ((*this)(p) != A(p)) {
            return false;
        } 
    }   

    return true; 
}   

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_equals(const Array2D &A) const {     
    chk_samesize_op(A,"element-wise equality");    

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) == A(p); 
    }   

    return B; 
}             

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_equals(const_reference val) const { 
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) == val;
    }

    return B; 
}   
        
template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_notequals(const Array2D &A) const {   
    chk_samesize_op(A,"element-wise inequality");

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) != A(p); 
    }   

    return B; 
}

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_notequals(const_reference val) const {
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) != val;
    }

    return B; 
}   

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_and(const Array2D &A) const { 
    chk_samesize_op(A,"element-wise 'and'");

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) && A(p);
    }

    return B; 
}   

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T, T_alloc>::this_or(const Array2D &A) const {    
    chk_samesize_op(A,"element-wise 'or'");     

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) || A(p);
    }

    return B;  
}

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_negate() const {  
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = !(*this)(p);
    }
    
    return B;
}

template <typename T, typename T_alloc> 
bool Array2D<T,T_alloc>::this_any_true() const {     
    for (difference_type p = 0; p < s; ++p) { 
        if ((*this)(p)) {
            return true;
        }
    }
    // returns false for empty arrays
    return false;
}

template <typename T, typename T_alloc>
bool Array2D<T,T_alloc>::this_all_true() const {     
    for (difference_type p = 0; p < s; ++p) { 
        if (!(*this)(p)) {
            return false;
        }
    }
    // returns true for empty arrays
    return true;
}

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::difference_type Array2D<T,T_alloc>::this_find(difference_type start) const {
    // Returns index of first true element starting from "start"
    for (difference_type p = start; p < s; ++p) { 
        if ((*this)(p)) {
            return p;
        }
    }
    // Returns -1 if value wasnt found
    return -1; 
}

// Relational operations -----------------------------------------------------//
template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_greaterthan(const Array2D &A) const {         
    chk_samesize_op(A,"element-wise greater-than");

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) > A(p);
    }   

    return B; 
}    

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_greaterthan(const_reference val) const {    
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) > val;
    }   

    return B; 
}        

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_greaterthanorequalto(const Array2D &A) const {    
    chk_samesize_op(A,"element-wise greater-than or equal to");     

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) >= A(p);
    }   

    return B; 
}   

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_greaterthanorequalto(const_reference val) const {  
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) >= val;
    }   

    return B; 
} 
        
template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_lessthan(const Array2D &A) const {     
    chk_samesize_op(A,"element-wise less-than");     

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) < A(p);
    }   

    return B; 
}
        
template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_lessthan(const_reference val) const {  
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) < val;
    }   

    return B; 
}

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_lessthanorequalto(const Array2D &A) const {   
    chk_samesize_op(A,"element-wise less-than or equal to");     

    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) <= A(p);
    }   

    return B; 
}  

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::bool_container Array2D<T,T_alloc>::this_lessthanorequalto(const_reference val) const {   
    bool_container B(h,w); 
    for (difference_type p = 0; p < s; ++p) { 
        B(p) = (*this)(p) <= val;
    }   

    return B; 
}  

// Arithmetic operations -----------------------------------------------------//
template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator+=(const Array2D &A) {   
    chk_samesize_op(A,"element-wise addition");
                    
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) += A(p);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator+=(const_reference val) {        
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) += val;
    }
    
    return *this;
}     

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator-=(const Array2D &A) {    
    chk_samesize_op(A,"element-wise subtraction");
    
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) -= A(p);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::this_flipsubassign(const Array2D &A) {
    chk_samesize_op(A,"element-wise subtraction");     

    for (difference_type p = 0; p < s; ++p) { 
        (*this)(p) = A(p)-(*this)(p); 
    }   

    return *this; 
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator-=(const_reference val) {        
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) -= val;
    }
    
    return *this;
}     

template <typename T, typename T_alloc>
Array2D<T,T_alloc>& Array2D<T,T_alloc>::this_flipsubassign(const_reference val) { 
    for (difference_type p = 0; p < s; ++p) { 
        (*this)(p) = val-(*this)(p); 
    }   

    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator*=(const Array2D &A) {   
    chk_samesize_op(A,"element-wise multiplication");
    
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) *= A(p);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator*=(const_reference val) {        
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) *= val;
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator/=(const Array2D &A) {    
    chk_samesize_op(A,"element-wise division");
    
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) /= A(p);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::this_flipdivassign(const Array2D &A) { 
    chk_samesize_op(A,"element-wise division");     

    for (difference_type p = 0; p < s; ++p) { 
        (*this)(p) = A(p)/(*this)(p); 
    }  

    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator/=(const_reference val) {        
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) /= val;
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::this_flipdivassign(const_reference val) { 
    for (difference_type p = 0; p < s; ++p) { 
        (*this)(p) = val/(*this)(p); 
    }    

    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator+() {        
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T,T_alloc>& Array2D<T,T_alloc>::operator-() {        
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) = -(*this)(p);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
Array2D<T, T_alloc>& Array2D<T,T_alloc>::this_sort() { 
    std::sort(ptr, ptr + s);
    
    return *this;
}

template <typename T, typename T_alloc> 
T Array2D<T,T_alloc>::this_sum() const {      
    value_type sum = value_type();
    for (difference_type p = 0; p < s; ++p) {
        sum += (*this)(p);
    }
    
    return sum;
}

template <typename T, typename T_alloc> 
T Array2D<T,T_alloc>::this_max() const {      
    chk_minsize_op(1,1,"max");
        
    return *std::max_element(ptr, ptr + s);
}

template <typename T, typename T_alloc> 
T Array2D<T,T_alloc>::this_min() const {      
    chk_minsize_op(1,1,"min");
    
    return *std::min_element(ptr, ptr + s);
}

template <typename T, typename T_alloc> 
T Array2D<T,T_alloc>::this_prctile(double percent) {      
    chk_minsize_op(1,1,"prctile");
    if (percent < 0 || percent > 1) {
        throw std::invalid_argument("Input percent of: " + std::to_string(percent) + " provided to prctile operator. Value must be between (inclusive) 0 and 1.");
    }
    
    // Sort, then return floored element closest to percent. No interpolation is
    // done.
    std::sort(ptr, ptr + s);
    
    return (*this)((s-1) * percent);
}

template <typename T, typename T_alloc> 
template<typename T_output> 
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T,T_alloc>::this_sqrt() {      
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) = std::sqrt((*this)(p));
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
template<typename T_output> 
typename std::enable_if<std::is_arithmetic<T>::value, T_output>::type Array2D<T,T_alloc>::this_pow(double n) {      
    for (difference_type p = 0; p < s; ++p) {
        (*this)(p) = std::pow((*this)(p),n);
    }
    
    return *this;
}

template <typename T, typename T_alloc> 
template <typename T_output> 
typename std::enable_if<!std::is_same<T,double>::value, T_output>::type  Array2D<T,T_alloc>::this_mat_mult(const Array2D<T,T_alloc> &A) const {    
    chk_mult_size(A);

    // Note this algorithm performs C = AB+C. Must ensure C is initialized 
    // to zero which is true when C is value initialized for numeric types.
    Array2D B(h, A.w);
    for (difference_type p = 0; p < A.w; ++p) {
        for (difference_type p2 = 0; p2 < w; ++p2) {
            for (difference_type p1 = 0; p1 < h; ++p1) {
                B(p1,p) += (*this)(p1,p2) * A(p2,p);
            }
        }
    }    

    return B;
}

template <typename T, typename T_alloc> 
template <typename T_output> 
typename std::enable_if<std::is_same<T,double>::value, T_output>::type Array2D<T,T_alloc>::this_mat_mult(const Array2D<T,T_alloc> &A) const {    
    return details::blas_mat_mult(*this, A);
}

// Additional arithmetic operations ------------------------------------------//
template <typename T, typename T_alloc> 
template <typename T_output> 
typename std::enable_if<std::is_floating_point<T>::value, T_output>::type Array2D<T,T_alloc>::this_dot(const Array2D &x) const {  
    chk_samesize_op(x,"dot product");     

    value_type sum = value_type();
    for (difference_type p = 0; p < s; ++p) {
        sum += (*this)(p)*x(p);
    }

    return sum;
}

template <typename T, typename T_alloc> 
template <typename T_output> 
typename std::enable_if<std::is_floating_point<T>::value, T_output>::type Array2D<T,T_alloc>::this_normalize() {      
    value_type norm = value_type();
    for (difference_type p = 0; p < s; ++p) {
        norm += std::pow((*this)(p),2.0);
    }
    norm = std::sqrt(norm);

    return (*this) /= norm;
}

// Utility Methods -----------------------------------------------------------// 
template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::coords Array2D<T,T_alloc>::get_range(const r_convert_1D &r) const {  
    // Get input inclusive range
    difference_type p1 = -1;
    difference_type p2 = -1;
    switch (r.range_type) {                      
        case r_convert::RANGE::SINGLE : 
            // Programmer error
            throw std::invalid_argument("Attempted to use 'single' element range for 1D range");  
        case r_convert::RANGE::LAST : 
            // Programmer error
            throw std::invalid_argument("Attempted to use 'last' element range for 1D range");  
        case r_convert::RANGE::COORDS : 
            p1 = r.p1;
            p2 = r.p2;
            break;
        case r_convert::RANGE::LAST_COORDS : 
            p1 = r.p1;
            p2 = s-1;
            break;
        case r_convert::RANGE::ALL :   
            p1 = 0;
            p2 = s-1;
            break;
    }
    
    // Test validity of coords and set them to "one past the end" range
    if (p2 < p1) {
        // Zero range - return {0,0} by default
        return {0,0};
    } else {
        // Positive range - test for validity
        if (p1 < 0 || p2 >= s) {
            throw std::invalid_argument("Attempted to perform 1D range type indexing using: ({" + std::to_string(p1) + "," + std::to_string(p2) + 
                                        "}) which is beyond the size of the array: " + size_string());           
        } else {
            // valid range
            return {p1,p2+1};
        }                
    }
} 

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::coords Array2D<T,T_alloc>::get_range(const r_convert_2D_1 &r) const {      
    // Get input inclusive range
    difference_type p1 = -1;
    difference_type p2 = -1;
    switch (r.range_type) {                 
        case r_convert::RANGE::SINGLE : 
            p1 = r.p1;
            p2 = r.p1;
            break;
        case r_convert::RANGE::LAST : 
            p1 = h-1;
            p2 = h-1;
            break;
        case r_convert::RANGE::COORDS : 
            p1 = r.p1;
            p2 = r.p2;
            break;
        case r_convert::RANGE::LAST_COORDS : 
            p1 = r.p1;
            p2 = h-1;
            break;
        case r_convert::RANGE::ALL :   
            p1 = 0;
            p2 = h-1;
            break;
    }
    
    // Test validity of coords and set them to "one past the end" range
    if (p2 < p1) {
        // Zero range - return {0,0} by default
        return {0,0};
    } else {
        // Positive range - test for validity
        if (p1 < 0 || p2 >= h) {
            throw std::invalid_argument("Attempted to perform 2D range type indexing using: ({" + std::to_string(p1) + "," + std::to_string(p2) + 
                                        "},) which is beyond the size of the Array: " + size_2D_string() + ".");           
        } else {
            // valid range
            return {p1,p2+1};
        }                
    }
} 

template <typename T, typename T_alloc>
typename Array2D<T,T_alloc>::coords Array2D<T,T_alloc>::get_range(const r_convert_2D_2 &r) const {
    // Get input inclusive range
    difference_type p1 = -1;
    difference_type p2 = -1;
    switch (r.range_type) {                 
        case r_convert::RANGE::SINGLE : 
            p1 = r.p1;
            p2 = r.p1;
            break;
        case r_convert::RANGE::LAST : 
            p1 = w-1;
            p2 = w-1;
            break;
        case r_convert::RANGE::COORDS : 
            p1 = r.p1;
            p2 = r.p2;
            break;
        case r_convert::RANGE::LAST_COORDS : 
            p1 = r.p1;
            p2 = w-1;
            break;
        case r_convert::RANGE::ALL :   
            p1 = 0;
            p2 = w-1;
            break;
    }
    
    // Test validity of coords and set them to "one past the end" range
    if (p2 < p1) {
        // Zero range - return {0,0} by default
        return {0,0};
    } else {
        // Positive range - test for validity
        if (p1 < 0 || p2 >= w) {
            throw std::invalid_argument("Attempted to perform 2D range type indexing using: (,{" + std::to_string(p1) + "," + std::to_string(p2) + 
                                        "}) which is beyond the size of the Array: " + size_2D_string() + ".");           
        } else {
            // valid range
            return {p1,p2+1};
        }                
    }
}     

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::pointer Array2D<T,T_alloc>::allocate(difference_type s_init) {
    pointer ptr_new;
    try {
        ptr_new = allocator_traits_type::allocate(alloc, s_init);
    } catch (std::bad_alloc &e) {
        // Bad alloc doesn't have a what() message, so just add one to terminal
        std::cerr << "---------------------------------------------" << std::endl 
                  << "Error: failed to allocate memory for Array2D." << std::endl
                  << "---------------------------------------------" << std::endl;
        throw;
    }

    return ptr_new;
}

template <typename T, typename T_alloc> 
typename Array2D<T,T_alloc>::pointer Array2D<T,T_alloc>::allocate_and_init(difference_type s_init, const_reference val) {
    pointer ptr_new = allocate(s_init);
    std::uninitialized_fill_n(ptr_new, s_init, val);
    
    return ptr_new;
}

template <typename T, typename T_alloc> 
template <typename T_it> 
typename Array2D<T,T_alloc>::pointer Array2D<T,T_alloc>::allocate_and_copy(difference_type s_init, T_it it) {    
    pointer ptr_new = allocate(s_init);
    std::uninitialized_copy_n(it, s_init, ptr_new);
        
    return ptr_new;
}

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::destroy_and_deallocate() {
    // Unlike free, cannot pass a null pointer to allocator, so test for it.
    // Note that r-values, after they are copied, are nullified through "make_null()"
    // method, which sets ptr to nullptr.
    if (ptr) {
        // Must destroy objects first before deallocating memory
        for (pointer ptr_delete = ptr + s; ptr_delete != ptr; /* empty */) {
            allocator_traits_type::destroy(alloc, --ptr_delete);
        }
        allocator_traits_type::deallocate(alloc, ptr, s);
    }
}

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_size_op(difference_type h, difference_type w, const std::string &op) const {
    if (this->h != h || this->w != w) {
        throw std::invalid_argument("Attempted to use " + op + " operator on array of size " + size_2D_string() + 
                                    ". Array must have size of (" + std::to_string(h) + "," + std::to_string(w) + ").");  
    }
}

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_samesize_op(const Array2D &A, const std::string &op) const {             
    if (!same_size(A)) { 
        throw std::invalid_argument("Attempted to use " + op + " operator on array of size " + size_2D_string() + 
                                    " with array of size " + A.size_2D_string() + ". Arrays must have the same size.");  
    }
}         

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_minsize_op(difference_type h, difference_type w, const std::string &op) const { 
    if (this->h < h || this->w < w) {
        throw std::invalid_argument("Attempted to use " + op + " operator on array of size " + size_2D_string() + 
                                    ". Array must be of size (" + std::to_string(h) + "," + std::to_string(w) + ") or greater.");  
    }
}      

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_column_op(const std::string &op) const { 
    if (this->w != 1) {
        throw std::invalid_argument("Attempted to use " + op + " operator on array of size " + size_2D_string() + 
                                    ". Array must be a column.");  
    }
}   

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_square_op(const std::string &op) const { 
    if (this->h != this->w) {
        throw std::invalid_argument("Attempted to use " + op + " operator on array of size " + size_2D_string() + 
                                    ". Array must be square.");  
    }
}  

template <typename T, typename T_alloc>
void Array2D<T,T_alloc>::chk_in_bounds_op(difference_type p, const std::string &op) const { 
    if (!in_bounds(p)) {
        throw std::invalid_argument("Attempted to use " + op + " with input of " + std::to_string(p) + " on array of size " + 
                                    size_string() + ".");  
    }
}

template <typename T, typename T_alloc>
void Array2D<T,T_alloc>::chk_in_bounds_op(difference_type p1, difference_type p2, const std::string &op) const { 
    if (!in_bounds(p1,p2)) {
        throw std::invalid_argument("Attempted to use " + op + " with input of (" + std::to_string(p1) + "," + std::to_string(p2) + ") on array of size " + 
                                    size_2D_string() + ".");  
    }
}

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_mult_size(const Array2D &A) const {    
    if (this->w != A.h) {
        throw std::invalid_argument("Attempted to multiply matrix of size " + size_2D_string() + " with " + A.size_2D_string() + 
                                    ". These sizes are incompatible for matrix multiplication.");
    }
}

template <typename T, typename T_alloc> 
void Array2D<T,T_alloc>::chk_kernel_size(const Array2D &kernel) const {    
    // Kernel must have odd dimensions and be smaller than or equal to A
    if (!(kernel.h % 2) || !(kernel.w % 2) || kernel.w > this->w || kernel.h > this->h) {
        throw std::invalid_argument("Attempted to convolve matrix of size " + size_2D_string() + " with kernel of size " + kernel.size_2D_string() + 
                                    ". Kernel must have odd dimensions and be equal to or smaller than matrix.");
    }
}

namespace details {    
    // Base Iterator ---------------------------------------------------------//
    template <typename T_container> 
    typename base_iterator<T_container>::reference base_iterator<T_container>::operator*() const {
        chk_in_range();

        return (*A_ptr)(p);
    }
    
    template <typename T_container> 
    void base_iterator<T_container>::chk_valid_increment() const {
        if (p >= A_ptr->size()) {
            throw std::invalid_argument("Attempted to increment Array2D iterator beyond last element.");
        }
    } 
    
    template <typename T_container> 
    void base_iterator<T_container>::chk_valid_decrement() const {
        if (p <= 0) {
            throw std::invalid_argument("Attempted to decrement Array2D iterator beyond first element.");
        }
    } 
        
    template <typename T_container> 
    void base_iterator<T_container>::chk_in_range() const {
        if (!A_ptr->in_bounds(p)) {
            throw std::invalid_argument("Attempted to dereference Array2D iterator out of range.");
        }
    } 
    
    // Simple Iterator -------------------------------------------------------//
    template <typename T_container> 
    inline simple_iterator<T_container>& simple_iterator<T_container>::operator++() {
        this->chk_valid_increment();

        ++this->p;

        return *this;
    }   

    template <typename T_container> 
    inline simple_iterator<T_container>& simple_iterator<T_container>::operator--() {
        this->chk_valid_decrement();

        --this->p;

        return *this;
    } 
    
    // Sub Iterator ----------------------------------------------------------//
    template <typename T_container> 
    sub_iterator<T_container>::sub_iterator(container &A, const coords &p_2D, const coords &r_sub1_2D, const coords &r_sub2_2D) : 
        base_iterator<container>(A,A.sub2ind(p_2D.first,p_2D.second)),  
        r_sub1_2D(r_sub1_2D),
        r_sub2_2D(r_sub2_2D),
        sub_p(p_2D.first-r_sub1_2D.first + (p_2D.second-r_sub2_2D.first)*(r_sub1_2D.second-r_sub1_2D.first)), 
        sub_h(r_sub1_2D.second-r_sub1_2D.first), 
        sub_w(r_sub2_2D.second-r_sub2_2D.first),
        sub_s((r_sub1_2D.second-r_sub1_2D.first) * (r_sub2_2D.second-r_sub2_2D.first)) {
        // Input ranges must be valid (check the criterion in the function))
        chk_valid_ranges();
    }
    
    template <typename T_container> 
    inline sub_iterator<T_container>& sub_iterator<T_container>::operator++() {
        this->chk_valid_increment();

        // increment sub_p, then determine position from it
        ++sub_p;
        this->p = this->A_ptr->sub2ind(sub_p % sub_h + r_sub1_2D.first, sub_p / sub_h + r_sub2_2D.first);

        return *this;
    }   

    template <typename T_container> 
    inline sub_iterator<T_container>& sub_iterator<T_container>::operator--() {
        this->chk_valid_decrement();

        // decrement sub_p, then determine position from it
        --sub_p;
        this->p = this->A_ptr->sub2ind(sub_p % sub_h + r_sub1_2D.first, sub_p / sub_h + r_sub2_2D.first);

        return *this;
    } 
    
    template <typename T_container> 
    void sub_iterator<T_container>::chk_valid_ranges() const {
        if (r_sub1_2D.first > r_sub1_2D.second || r_sub1_2D.first < 0 || r_sub1_2D.second > this->A_ptr->height() ||
            r_sub2_2D.first > r_sub2_2D.second || r_sub2_2D.first < 0 || r_sub2_2D.second > this->A_ptr->width()) {
            throw std::invalid_argument("Range of ({" + std::to_string(r_sub1_2D.first) + "," + std::to_string(r_sub1_2D.second-1) + "},{" +
                                                        std::to_string(r_sub2_2D.first) + "," + std::to_string(r_sub2_2D.second-1) + "}) is not valid for sub iterator with size of: " + 
                                        this->A_ptr->size_2D_string() + ".");
        }
    } 
    
    // Bool Iterator ---------------------------------------------------------//        
    template <typename T_container> 
    bool_iterator<T_container>::bool_iterator(container &A, difference_type p, const bool_container *A_bool_ptr) : 
        base_iterator<container>(A,p), A_bool_ptr(A_bool_ptr) {
        chk_same_size();
        // If the initial position is (0,0), check to see if it is true; if not,
        // iterate until the first true position is found. This is done for 
        // convenience so the begin() iterator can be set with a position of 0 
        // by the caller.
        if (this->p == 0  && !this->A_ptr->empty() && !(*this->A_bool_ptr)(0)) {
            ++(*this);
        }
    }  
    
    template <typename T_container> 
    inline bool_iterator<T_container>& bool_iterator<T_container>::operator++() {
        this->chk_valid_increment();
        
        // increment p until true value is found
        while (++this->p < this->A_ptr->size()) {        
            if ((*A_bool_ptr)(this->p)) {
                return *this;
            }
        }
        return *this;
    }   

    template <typename T_container> 
    inline bool_iterator<T_container>& bool_iterator<T_container>::operator--() {
        this->chk_valid_decrement();

        // decrement p until true value is found
        while (this->p > 0) {        
            if ((*A_bool_ptr)(--this->p)) {
                return *this;
            }
        } 
        
        return *this;
    } 
        
    template <typename T_container> 
    void bool_iterator<T_container>::chk_same_size() const {
        if (!this->A_ptr->same_size(*A_bool_ptr)) {
            throw std::invalid_argument("Attempted to use boolean Array of size: " + A_bool_ptr->size_2D_string() + 
                                        " to form iterator for Array2D of size: " + this->A_ptr->size_2D_string() + ".");
        }
    }       
    
    // Base Region -----------------------------------------------------------//
    template <typename T_container>
    base_region<T_container>& base_region<T_container>::operator=(const base_region &reg) {
        if (this->region_h != reg.region_h || this->region_w != reg.region_w) {
            throw std::invalid_argument("Attempted to assign region of size: " + reg.region_size_2D_string() + 
                                        " to region of size: " + region_size_2D_string() + ".");
        }
               
        std::copy(reg.begin(), reg.end(), this->begin());
        
        return *this;
    }  
    
    template <typename T_container>
    template <typename T_container2>
    typename std::enable_if<std::is_same<typename container_traits<T_container2>::nonconst_container, typename base_region<T_container>::nonconst_container>::value, base_region<T_container>&>::type base_region<T_container>::operator=(const base_region<T_container2> &reg) {
        if (this->region_h != reg.region_h || this->region_w != reg.region_w) {
            throw std::invalid_argument("Attempted to assign region of size: " + reg.region_size_2D_string() + 
                                        " to region of size: " + region_size_2D_string() + ".");
        }
               
        std::copy(reg.begin(), reg.end(), this->begin());
        
        return *this;
    }  
        
    template <typename T_container>
    base_region<T_container>& base_region<T_container>::operator=(const_container &A) {
        if (this->region_h != A.height() || this->region_w != A.width()) {
            throw std::invalid_argument("Attempted to assign Array2D of size: " + A.size_2D_string() + 
                                        " to region of size: " + region_size_2D_string() + ".");
        }
        
        std::copy(A.begin(), A.end(), this->begin());
        
        return *this;
    }  
        
    template <typename T_container>
    base_region<T_container>& base_region<T_container>::operator=(const_reference val) {        
        std::fill(this->begin(), this->end(), val);
        
        return *this;
    }  
    
    // Simple Region ---------------------------------------------------------//
    template <typename T_container>
    simple_region<T_container>::simple_region(container &A, const coords &r) : base_region<container>(A,r.second-r.first,1), r(r) {
        // Input ranges must be valid (check the criterion in the function))
        chk_valid_range();
    }  
    
    template <typename T_container>
    void simple_region<T_container>::chk_valid_range() const {
     if (r.first > r.second || r.first < 0 || r.second > this->A_ptr->size()) {
            throw std::invalid_argument("Range of: ({" + std::to_string(r.first) + "," + std::to_string(r.second-1) + 
                                        "}) is not valid for region with max size of: " + this->A_ptr->size_string() + ".");
        }
    }  
            
    // Sub Region ------------------------------------------------------------//
    template <typename T_container>     
    sub_region<T_container>::sub_region(container &A, const coords &r_sub1_2D, const coords &r_sub2_2D) : 
        base_region<container>(A,r_sub1_2D.second-r_sub1_2D.first,r_sub2_2D.second-r_sub2_2D.first),
        r_sub1_2D(r_sub1_2D),
        r_sub2_2D(r_sub2_2D),
        sub_h(r_sub1_2D.second-r_sub1_2D.first), 
        sub_w(r_sub2_2D.second-r_sub2_2D.first),
        sub_s((r_sub1_2D.second-r_sub1_2D.first) * (r_sub2_2D.second-r_sub2_2D.first)) {
        // Input ranges must be valid (check the criterion in the function))
        chk_valid_ranges();
    } 
    
    template <typename T_container> 
    void sub_region<T_container>::chk_valid_ranges() const {
        if (r_sub1_2D.first > r_sub1_2D.second || r_sub1_2D.first < 0 || r_sub1_2D.second > this->A_ptr->height() ||
            r_sub2_2D.first > r_sub2_2D.second || r_sub2_2D.first < 0 || r_sub2_2D.second > this->A_ptr->width()) {
            throw std::invalid_argument("Range of ({" + std::to_string(r_sub1_2D.first) + "," + std::to_string(r_sub1_2D.second-1) + "},{" +
                                                        std::to_string(r_sub2_2D.first) + "," + std::to_string(r_sub2_2D.second-1) + "}) is not valid for sub region with max size of: " + 
                                        this->A_ptr->size_2D_string() + ".");
        }
    } 
    
    // Bool Region -----------------------------------------------------------//    
    template <typename T_container>    
    bool_region<T_container>::bool_region(container &A, bool_container A_bool) : 
        base_region<container>(A,sum(convert(A_bool,difference_type())),1), A_bool_ptr(std::make_shared<bool_container>(std::move(A_bool))) {
        chk_same_size();
    } 
        
    template <typename T_container>    
    void bool_region<T_container>::chk_same_size() const {
        if (!this->A_ptr->same_size(*A_bool_ptr)) {
            throw std::invalid_argument("Attempted to use boolean Array of size: " + A_bool_ptr->size_2D_string() + 
                                        " to index Array2D of size: " + this->A_ptr->size_2D_string() + ".");
        }
    } 
      
    // Nearest Interpolator --------------------------------------------------//
    template <typename T_container> 
    inline typename nearest_interp<T_container>::value_type nearest_interp<T_container>::operator()(double p1, double p2) const { 
        if (out_of_bounds(p1, p2)) {
            return std::numeric_limits<value_type>::quiet_NaN();
        }

        return (*this->A_ptr)(std::round(p1),std::round(p2));
    }
    
    template <typename T_container> 
    inline typename nearest_interp<T_container>::const_container& nearest_interp<T_container>::first_order(double p1, double p2) const { 
        if (out_of_bounds(p1, p2)) {
            this->first_order_buf(0) = this->first_order_buf(1) = this->first_order_buf(2) = std::numeric_limits<value_type>::quiet_NaN();
            return this->first_order_buf;
        }
        
        this->first_order_buf(0) = (*this->A_ptr)(std::round(p1),std::round(p2));
        this->first_order_buf(1) = this->first_order_buf(2) = 0; // Gradients are zero

        return this->first_order_buf;
    }    
    
    // Bilinear Interpolator -------------------------------------------------//    
    template <typename T_container> 
    inline typename linear_interp<T_container>::value_type linear_interp<T_container>::operator()(double p1, double p2) const { 
        if (out_of_bounds(p1,p2)) {
            return std::numeric_limits<value_type>::quiet_NaN();
        }

        double delta_p1 = p1 - std::floor(p1);
        double delta_p2 = p2 - std::floor(p2);
        
        return (*this->A_ptr)(p1,p2) * (1-delta_p1) * (1-delta_p2) +
               (*this->A_ptr)(p1+1,p2) * delta_p1 * (1-delta_p2) +
               (*this->A_ptr)(p1,p2+1) * (1-delta_p1) * delta_p2 +
               (*this->A_ptr)(p1+1,p2+1) * delta_p1 * delta_p2;
    }    
    
    template <typename T_container> 
    inline typename linear_interp<T_container>::const_container& linear_interp<T_container>::first_order(double p1, double p2) const { 
        if (out_of_bounds(p1,p2)) {
            this->first_order_buf(0) = this->first_order_buf(1) = this->first_order_buf(2) = std::numeric_limits<value_type>::quiet_NaN();
            return this->first_order_buf;
        }
       
        double delta_p1 = p1 - std::floor(p1);
        double delta_p2 = p2 - std::floor(p2);
                
        this->first_order_buf(0) = (*this->A_ptr)(p1,p2) * (1-delta_p1) * (1-delta_p2) +
                                   (*this->A_ptr)(p1+1,p2) * delta_p1 * (1-delta_p2) +
                                   (*this->A_ptr)(p1,p2+1) * (1-delta_p1) * delta_p2 +
                                   (*this->A_ptr)(p1+1,p2+1) * delta_p1 * delta_p2;
        
        this->first_order_buf(1) = (*this->A_ptr)(p1,p2) * (-1) * (1-delta_p2) +
                                   (*this->A_ptr)(p1+1,p2) * 1 * (1-delta_p2) +
                                   (*this->A_ptr)(p1,p2+1) * (-1) * delta_p2 +
                                   (*this->A_ptr)(p1+1,p2+1) * 1 * delta_p2;
        
        this->first_order_buf(2) = (*this->A_ptr)(p1,p2) * (1-delta_p1) * (-1) +
                                   (*this->A_ptr)(p1+1,p2) * delta_p1 * (-1) +
                                   (*this->A_ptr)(p1,p2+1) * (1-delta_p1) * 1 +
                                   (*this->A_ptr)(p1+1,p2+1) * delta_p1 * 1;
                
        return this->first_order_buf;
    } 
    
    // Coefficient Matrix Interpolator ---------------------------------------//
    template <typename T_container> 
    inline typename coef_mat_interp_base<T_container>::value_type coef_mat_interp_base<T_container>::operator()(double p1, double p2) const {
        // This is the general interpolation scheme for interpolators that use
        // a coefficient matrix. Can be overridden in child classes to increase
        // speed.
        if (this->out_of_bounds(p1,p2)) {
            return std::numeric_limits<value_type>::quiet_NaN();
        }
                
        // Get powers of delta_p1
        p1_pow_buf = get_p_pow(p1_pow_buf, p1 - std::floor(p1));

        // Get powers of delta_p2
        p2_pow_buf = get_p_pow(p2_pow_buf, p2 - std::floor(p2));

        // Get coefficient matrix
        const auto &coef_mat = this->get_coef_mat(p1, p2);
        
        // Interpolate value        
        return t_vec_mat_vec(p1_pow_buf, coef_mat, p2_pow_buf);
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
            
    // Bicubic Interpolator --------------------------------------------------//       
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
    
    // Precomputed Bicubic Interpolator --------------------------------------//
    template <typename T_container> 
    cubic_interp_precompute<T_container>::cubic_interp_precompute(const_container &A) : cubic_interp_base<container>(A) { 
        // pre-allocate memory for entire coefficient array - note that this will 
        // be a lot of memory - 16 times the size of the original array.
        coef_mat_precompute_ptr = std::make_shared<Array2D<container>>(this->A_ptr->height(), this->A_ptr->width(), container(4,4));

        for (difference_type p2 = 1; p2 < this->A_ptr->width() - 2; ++p2) {
            for (difference_type p1 = 1; p1 < this->A_ptr->height() - 2; ++p1) {
                this->calc_coef_mat((*coef_mat_precompute_ptr)(p1,p2), *this->A_ptr, p1 - 1, p2 - 1);
            }
        }
    } 
        
    // Biquintic B-spline base class -----------------------------------------//
    template <typename T_container> 
    inline typename quintic_interp_base<T_container>::value_type quintic_interp_base<T_container>::operator()(double p1, double p2) const {        
        if (out_of_bounds(p1,p2)) {
            return std::numeric_limits<value_type>::quiet_NaN();
        }

        double delta_p1 = p1 - std::floor(p1);
        double delta_p2 = p2 - std::floor(p2);

        this->p1_pow_buf(0) = 1.0;
        this->p1_pow_buf(1) = delta_p1;
        this->p1_pow_buf(2) = this->p1_pow_buf(1)*delta_p1;
        this->p1_pow_buf(3) = this->p1_pow_buf(2)*delta_p1;
        this->p1_pow_buf(4) = this->p1_pow_buf(3)*delta_p1;
        this->p1_pow_buf(5) = this->p1_pow_buf(4)*delta_p1;

        this->p2_pow_buf(0) = 1.0;
        this->p2_pow_buf(1) = delta_p2;
        this->p2_pow_buf(2) = this->p2_pow_buf(1)*delta_p2;
        this->p2_pow_buf(3) = this->p2_pow_buf(2)*delta_p2;
        this->p2_pow_buf(4) = this->p2_pow_buf(3)*delta_p2;
        this->p2_pow_buf(5) = this->p2_pow_buf(4)*delta_p2;

        const auto &coef_mat = this->get_coef_mat(p1, p2);
        
        return (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3)+this->p2_pow_buf(4)*coef_mat(0,4)+this->p2_pow_buf(5)*coef_mat(0,5))*this->p1_pow_buf(0)+
               (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3)+this->p2_pow_buf(4)*coef_mat(1,4)+this->p2_pow_buf(5)*coef_mat(1,5))*this->p1_pow_buf(1)+
               (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3)+this->p2_pow_buf(4)*coef_mat(2,4)+this->p2_pow_buf(5)*coef_mat(2,5))*this->p1_pow_buf(2)+
               (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3)+this->p2_pow_buf(4)*coef_mat(3,4)+this->p2_pow_buf(5)*coef_mat(3,5))*this->p1_pow_buf(3)+
               (this->p2_pow_buf(0)*coef_mat(4,0)+this->p2_pow_buf(1)*coef_mat(4,1)+this->p2_pow_buf(2)*coef_mat(4,2)+this->p2_pow_buf(3)*coef_mat(4,3)+this->p2_pow_buf(4)*coef_mat(4,4)+this->p2_pow_buf(5)*coef_mat(4,5))*this->p1_pow_buf(4)+
               (this->p2_pow_buf(0)*coef_mat(5,0)+this->p2_pow_buf(1)*coef_mat(5,1)+this->p2_pow_buf(2)*coef_mat(5,2)+this->p2_pow_buf(3)*coef_mat(5,3)+this->p2_pow_buf(4)*coef_mat(5,4)+this->p2_pow_buf(5)*coef_mat(5,5))*this->p1_pow_buf(5);
    }
   
    template <typename T_container> 
    inline typename quintic_interp_base<T_container>::const_container& quintic_interp_base<T_container>::first_order(double p1, double p2) const { 
        if (out_of_bounds(p1,p2)) {
            this->first_order_buf(0) = this->first_order_buf(1) = this->first_order_buf(2) = std::numeric_limits<value_type>::quiet_NaN();
            return this->first_order_buf;
        }
        
        double delta_p1 = p1 - std::floor(p1);
        double delta_p2 = p2 - std::floor(p2);
                
        this->p1_pow_buf(0) = 1.0;
        this->p1_pow_buf(1) = delta_p1;
        this->p1_pow_buf(2) = this->p1_pow_buf(1)*delta_p1;
        this->p1_pow_buf(3) = this->p1_pow_buf(2)*delta_p1;
        this->p1_pow_buf(4) = this->p1_pow_buf(3)*delta_p1;
        this->p1_pow_buf(5) = this->p1_pow_buf(4)*delta_p1;
        
        this->p1_pow_dp1_buf(0) = 0.0;
        this->p1_pow_dp1_buf(1) = 1.0;
        this->p1_pow_dp1_buf(2) = 2.0 * this->p1_pow_buf(1);
        this->p1_pow_dp1_buf(3) = 3.0 * this->p1_pow_buf(2);
        this->p1_pow_dp1_buf(4) = 4.0 * this->p1_pow_buf(3);
        this->p1_pow_dp1_buf(5) = 5.0 * this->p1_pow_buf(4);
        
        this->p2_pow_buf(0) = 1.0;
        this->p2_pow_buf(1) = delta_p2;
        this->p2_pow_buf(2) = this->p2_pow_buf(1)*delta_p2;
        this->p2_pow_buf(3) = this->p2_pow_buf(2)*delta_p2;
        this->p2_pow_buf(4) = this->p2_pow_buf(3)*delta_p2;
        this->p2_pow_buf(5) = this->p2_pow_buf(4)*delta_p2;
        
        this->p2_pow_dp2_buf(0) = 0.0;
        this->p2_pow_dp2_buf(1) = 1.0;
        this->p2_pow_dp2_buf(2) = 2.0 * this->p2_pow_buf(1);
        this->p2_pow_dp2_buf(3) = 3.0 * this->p2_pow_buf(2);
        this->p2_pow_dp2_buf(4) = 4.0 * this->p2_pow_buf(3);
        this->p2_pow_dp2_buf(5) = 5.0 * this->p2_pow_buf(4);
        
        const auto &coef_mat = this->get_coef_mat(p1, p2);
        
        this->first_order_buf(0) = (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3)+this->p2_pow_buf(4)*coef_mat(0,4)+this->p2_pow_buf(5)*coef_mat(0,5))*this->p1_pow_buf(0)+
                                   (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3)+this->p2_pow_buf(4)*coef_mat(1,4)+this->p2_pow_buf(5)*coef_mat(1,5))*this->p1_pow_buf(1)+
                                   (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3)+this->p2_pow_buf(4)*coef_mat(2,4)+this->p2_pow_buf(5)*coef_mat(2,5))*this->p1_pow_buf(2)+
                                   (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3)+this->p2_pow_buf(4)*coef_mat(3,4)+this->p2_pow_buf(5)*coef_mat(3,5))*this->p1_pow_buf(3)+
                                   (this->p2_pow_buf(0)*coef_mat(4,0)+this->p2_pow_buf(1)*coef_mat(4,1)+this->p2_pow_buf(2)*coef_mat(4,2)+this->p2_pow_buf(3)*coef_mat(4,3)+this->p2_pow_buf(4)*coef_mat(4,4)+this->p2_pow_buf(5)*coef_mat(4,5))*this->p1_pow_buf(4)+
                                   (this->p2_pow_buf(0)*coef_mat(5,0)+this->p2_pow_buf(1)*coef_mat(5,1)+this->p2_pow_buf(2)*coef_mat(5,2)+this->p2_pow_buf(3)*coef_mat(5,3)+this->p2_pow_buf(4)*coef_mat(5,4)+this->p2_pow_buf(5)*coef_mat(5,5))*this->p1_pow_buf(5);
        
        this->first_order_buf(1) = (this->p2_pow_buf(0)*coef_mat(0,0)+this->p2_pow_buf(1)*coef_mat(0,1)+this->p2_pow_buf(2)*coef_mat(0,2)+this->p2_pow_buf(3)*coef_mat(0,3)+this->p2_pow_buf(4)*coef_mat(0,4)+this->p2_pow_buf(5)*coef_mat(0,5))*this->p1_pow_dp1_buf(0)+
                                   (this->p2_pow_buf(0)*coef_mat(1,0)+this->p2_pow_buf(1)*coef_mat(1,1)+this->p2_pow_buf(2)*coef_mat(1,2)+this->p2_pow_buf(3)*coef_mat(1,3)+this->p2_pow_buf(4)*coef_mat(1,4)+this->p2_pow_buf(5)*coef_mat(1,5))*this->p1_pow_dp1_buf(1)+
                                   (this->p2_pow_buf(0)*coef_mat(2,0)+this->p2_pow_buf(1)*coef_mat(2,1)+this->p2_pow_buf(2)*coef_mat(2,2)+this->p2_pow_buf(3)*coef_mat(2,3)+this->p2_pow_buf(4)*coef_mat(2,4)+this->p2_pow_buf(5)*coef_mat(2,5))*this->p1_pow_dp1_buf(2)+
                                   (this->p2_pow_buf(0)*coef_mat(3,0)+this->p2_pow_buf(1)*coef_mat(3,1)+this->p2_pow_buf(2)*coef_mat(3,2)+this->p2_pow_buf(3)*coef_mat(3,3)+this->p2_pow_buf(4)*coef_mat(3,4)+this->p2_pow_buf(5)*coef_mat(3,5))*this->p1_pow_dp1_buf(3)+
                                   (this->p2_pow_buf(0)*coef_mat(4,0)+this->p2_pow_buf(1)*coef_mat(4,1)+this->p2_pow_buf(2)*coef_mat(4,2)+this->p2_pow_buf(3)*coef_mat(4,3)+this->p2_pow_buf(4)*coef_mat(4,4)+this->p2_pow_buf(5)*coef_mat(4,5))*this->p1_pow_dp1_buf(4)+
                                   (this->p2_pow_buf(0)*coef_mat(5,0)+this->p2_pow_buf(1)*coef_mat(5,1)+this->p2_pow_buf(2)*coef_mat(5,2)+this->p2_pow_buf(3)*coef_mat(5,3)+this->p2_pow_buf(4)*coef_mat(5,4)+this->p2_pow_buf(5)*coef_mat(5,5))*this->p1_pow_dp1_buf(5);
        
        this->first_order_buf(2) = (this->p2_pow_dp2_buf(0)*coef_mat(0,0)+this->p2_pow_dp2_buf(1)*coef_mat(0,1)+this->p2_pow_dp2_buf(2)*coef_mat(0,2)+this->p2_pow_dp2_buf(3)*coef_mat(0,3)+this->p2_pow_dp2_buf(4)*coef_mat(0,4)+this->p2_pow_dp2_buf(5)*coef_mat(0,5))*this->p1_pow_buf(0)+
                                   (this->p2_pow_dp2_buf(0)*coef_mat(1,0)+this->p2_pow_dp2_buf(1)*coef_mat(1,1)+this->p2_pow_dp2_buf(2)*coef_mat(1,2)+this->p2_pow_dp2_buf(3)*coef_mat(1,3)+this->p2_pow_dp2_buf(4)*coef_mat(1,4)+this->p2_pow_dp2_buf(5)*coef_mat(1,5))*this->p1_pow_buf(1)+
                                   (this->p2_pow_dp2_buf(0)*coef_mat(2,0)+this->p2_pow_dp2_buf(1)*coef_mat(2,1)+this->p2_pow_dp2_buf(2)*coef_mat(2,2)+this->p2_pow_dp2_buf(3)*coef_mat(2,3)+this->p2_pow_dp2_buf(4)*coef_mat(2,4)+this->p2_pow_dp2_buf(5)*coef_mat(2,5))*this->p1_pow_buf(2)+
                                   (this->p2_pow_dp2_buf(0)*coef_mat(3,0)+this->p2_pow_dp2_buf(1)*coef_mat(3,1)+this->p2_pow_dp2_buf(2)*coef_mat(3,2)+this->p2_pow_dp2_buf(3)*coef_mat(3,3)+this->p2_pow_dp2_buf(4)*coef_mat(3,4)+this->p2_pow_dp2_buf(5)*coef_mat(3,5))*this->p1_pow_buf(3)+
                                   (this->p2_pow_dp2_buf(0)*coef_mat(4,0)+this->p2_pow_dp2_buf(1)*coef_mat(4,1)+this->p2_pow_dp2_buf(2)*coef_mat(4,2)+this->p2_pow_dp2_buf(3)*coef_mat(4,3)+this->p2_pow_dp2_buf(4)*coef_mat(4,4)+this->p2_pow_dp2_buf(5)*coef_mat(4,5))*this->p1_pow_buf(4)+
                                   (this->p2_pow_dp2_buf(0)*coef_mat(5,0)+this->p2_pow_dp2_buf(1)*coef_mat(5,1)+this->p2_pow_dp2_buf(2)*coef_mat(5,2)+this->p2_pow_dp2_buf(3)*coef_mat(5,3)+this->p2_pow_dp2_buf(4)*coef_mat(5,4)+this->p2_pow_dp2_buf(5)*coef_mat(5,5))*this->p1_pow_buf(5);
        
        return this->first_order_buf;
    }
    
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
        return std::make_shared<container>(deconv(deconv(pad(A, bcoef_border, PAD::EXPAND_EDGES), t(kernel_qb)), kernel_qb)); 
    }
    
    template <typename T_container> 
    inline typename quintic_interp_base<T_container>::const_container& quintic_interp_base<T_container>::calc_coef_mat(container &coef_mat_buf, const_container &bcoef, difference_type p1, difference_type p2) const {
        // p1 and p2 refer to the top-left corner of the desired coefficient matrix
        #ifndef NDEBUG
        if (p1 < 0 || p1+5 >= bcoef.height() || p2 < 0 || p2+5 >= bcoef.width()) {
            throw std::invalid_argument("p1 and p2 are outside range of array for calc_coef_mat() with quintic interpolation - this is a programmer error.");
        }
        #endif
        
        coef_mat_buf(0,0) = 0.00006944444444444444*bcoef(p1,p2)+0.001805555555555556*bcoef(p1+1,p2)+0.001805555555555556*bcoef(p1+4,p2+1)+0.004583333333333333*bcoef(p1,p2+2)+0.1191666666666667*bcoef(p1+1,p2+2)+0.3025*bcoef(p1+2,p2+2)+0.1191666666666667*bcoef(p1+3,p2+2)+0.004583333333333333*bcoef(p1+4,p2+2)+0.001805555555555556*bcoef(p1,p2+3)+0.04694444444444444*bcoef(p1+1,p2+3)+0.004583333333333333*bcoef(p1+2,p2)+0.1191666666666667*bcoef(p1+2,p2+3)+0.04694444444444444*bcoef(p1+3,p2+3)+0.001805555555555556*bcoef(p1+4,p2+3)+0.00006944444444444444*bcoef(p1,p2+4)+0.001805555555555556*bcoef(p1+1,p2+4)+0.004583333333333333*bcoef(p1+2,p2+4)+0.001805555555555556*bcoef(p1+3,p2+4)+0.00006944444444444444*bcoef(p1+4,p2+4)+0.001805555555555556*bcoef(p1+3,p2)+0.00006944444444444444*bcoef(p1+4,p2)+0.001805555555555556*bcoef(p1,p2+1)+0.04694444444444444*bcoef(p1+1,p2+1)+0.1191666666666667*bcoef(p1+2,p2+1)+0.04694444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(1,0) = 0.009027777777777778*bcoef(p1+4,p2+1)-0.003472222222222222*bcoef(p1+1,p2)-0.0003472222222222222*bcoef(p1,p2)-0.02291666666666667*bcoef(p1,p2+2)-0.2291666666666667*bcoef(p1+1,p2+2)+0.2291666666666667*bcoef(p1+3,p2+2)+0.02291666666666667*bcoef(p1+4,p2+2)-0.009027777777777778*bcoef(p1,p2+3)-0.09027777777777778*bcoef(p1+1,p2+3)+0.09027777777777778*bcoef(p1+3,p2+3)+0.009027777777777778*bcoef(p1+4,p2+3)-0.0003472222222222222*bcoef(p1,p2+4)-0.003472222222222222*bcoef(p1+1,p2+4)+0.003472222222222222*bcoef(p1+3,p2+4)+0.0003472222222222222*bcoef(p1+4,p2+4)+0.003472222222222222*bcoef(p1+3,p2)+0.0003472222222222222*bcoef(p1+4,p2)-0.009027777777777778*bcoef(p1,p2+1)-0.09027777777777778*bcoef(p1+1,p2+1)+0.09027777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(2,0) = 0.0006944444444444444*bcoef(p1,p2)+0.001388888888888889*bcoef(p1+1,p2)+0.01805555555555556*bcoef(p1+4,p2+1)+0.04583333333333333*bcoef(p1,p2+2)+0.09166666666666667*bcoef(p1+1,p2+2)-0.275*bcoef(p1+2,p2+2)+0.09166666666666667*bcoef(p1+3,p2+2)+0.04583333333333333*bcoef(p1+4,p2+2)+0.01805555555555556*bcoef(p1,p2+3)+0.03611111111111111*bcoef(p1+1,p2+3)-0.004166666666666667*bcoef(p1+2,p2)-0.1083333333333333*bcoef(p1+2,p2+3)+0.03611111111111111*bcoef(p1+3,p2+3)+0.01805555555555556*bcoef(p1+4,p2+3)+0.0006944444444444444*bcoef(p1,p2+4)+0.001388888888888889*bcoef(p1+1,p2+4)-0.004166666666666667*bcoef(p1+2,p2+4)+0.001388888888888889*bcoef(p1+3,p2+4)+0.0006944444444444444*bcoef(p1+4,p2+4)+0.001388888888888889*bcoef(p1+3,p2)+0.0006944444444444444*bcoef(p1+4,p2)+0.01805555555555556*bcoef(p1,p2+1)+0.03611111111111111*bcoef(p1+1,p2+1)-0.1083333333333333*bcoef(p1+2,p2+1)+0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(3,0) = 0.001388888888888889*bcoef(p1+1,p2)-0.0006944444444444444*bcoef(p1,p2)+0.01805555555555556*bcoef(p1+4,p2+1)-0.04583333333333333*bcoef(p1,p2+2)+0.09166666666666667*bcoef(p1+1,p2+2)-0.09166666666666667*bcoef(p1+3,p2+2)+0.04583333333333333*bcoef(p1+4,p2+2)-0.01805555555555556*bcoef(p1,p2+3)+0.03611111111111111*bcoef(p1+1,p2+3)-0.03611111111111111*bcoef(p1+3,p2+3)+0.01805555555555556*bcoef(p1+4,p2+3)-0.0006944444444444444*bcoef(p1,p2+4)+0.001388888888888889*bcoef(p1+1,p2+4)-0.001388888888888889*bcoef(p1+3,p2+4)+0.0006944444444444444*bcoef(p1+4,p2+4)-0.001388888888888889*bcoef(p1+3,p2)+0.0006944444444444444*bcoef(p1+4,p2)-0.01805555555555556*bcoef(p1,p2+1)+0.03611111111111111*bcoef(p1+1,p2+1)-0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(4,0) = 0.0003472222222222222*bcoef(p1,p2)-0.001388888888888889*bcoef(p1+1,p2)+0.009027777777777778*bcoef(p1+4,p2+1)+0.02291666666666667*bcoef(p1,p2+2)-0.09166666666666667*bcoef(p1+1,p2+2)+0.1375*bcoef(p1+2,p2+2)-0.09166666666666667*bcoef(p1+3,p2+2)+0.02291666666666667*bcoef(p1+4,p2+2)+0.009027777777777778*bcoef(p1,p2+3)-0.03611111111111111*bcoef(p1+1,p2+3)+0.002083333333333333*bcoef(p1+2,p2)+0.05416666666666667*bcoef(p1+2,p2+3)-0.03611111111111111*bcoef(p1+3,p2+3)+0.009027777777777778*bcoef(p1+4,p2+3)+0.0003472222222222222*bcoef(p1,p2+4)-0.001388888888888889*bcoef(p1+1,p2+4)+0.002083333333333333*bcoef(p1+2,p2+4)-0.001388888888888889*bcoef(p1+3,p2+4)+0.0003472222222222222*bcoef(p1+4,p2+4)-0.001388888888888889*bcoef(p1+3,p2)+0.0003472222222222222*bcoef(p1+4,p2)+0.009027777777777778*bcoef(p1,p2+1)-0.03611111111111111*bcoef(p1+1,p2+1)+0.05416666666666667*bcoef(p1+2,p2+1)-0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(5,0) = 0.0003472222222222222*bcoef(p1+1,p2)-0.00006944444444444444*bcoef(p1,p2)-0.009027777777777778*bcoef(p1+4,p2+1)+0.001805555555555556*bcoef(p1+5,p2+1)-0.004583333333333333*bcoef(p1,p2+2)+0.02291666666666667*bcoef(p1+1,p2+2)-0.04583333333333333*bcoef(p1+2,p2+2)+0.04583333333333333*bcoef(p1+3,p2+2)-0.02291666666666667*bcoef(p1+4,p2+2)+0.004583333333333333*bcoef(p1+5,p2+2)-0.001805555555555556*bcoef(p1,p2+3)+0.009027777777777778*bcoef(p1+1,p2+3)-0.0006944444444444444*bcoef(p1+2,p2)-0.01805555555555556*bcoef(p1+2,p2+3)+0.01805555555555556*bcoef(p1+3,p2+3)-0.009027777777777778*bcoef(p1+4,p2+3)+0.001805555555555556*bcoef(p1+5,p2+3)-0.00006944444444444444*bcoef(p1,p2+4)+0.0003472222222222222*bcoef(p1+1,p2+4)-0.0006944444444444444*bcoef(p1+2,p2+4)+0.0006944444444444444*bcoef(p1+3,p2+4)-0.0003472222222222222*bcoef(p1+4,p2+4)+0.00006944444444444444*bcoef(p1+5,p2+4)+0.0006944444444444444*bcoef(p1+3,p2)-0.0003472222222222222*bcoef(p1+4,p2)+0.00006944444444444444*bcoef(p1+5,p2)-0.001805555555555556*bcoef(p1,p2+1)+0.009027777777777778*bcoef(p1+1,p2+1)-0.01805555555555556*bcoef(p1+2,p2+1)+0.01805555555555556*bcoef(p1+3,p2+1);
        coef_mat_buf(0,1) = 0.003472222222222222*bcoef(p1,p2+3)-0.009027777777777778*bcoef(p1+1,p2)-0.003472222222222222*bcoef(p1+4,p2+1)-0.0003472222222222222*bcoef(p1,p2)+0.09027777777777778*bcoef(p1+1,p2+3)-0.02291666666666667*bcoef(p1+2,p2)+0.2291666666666667*bcoef(p1+2,p2+3)+0.09027777777777778*bcoef(p1+3,p2+3)+0.003472222222222222*bcoef(p1+4,p2+3)+0.0003472222222222222*bcoef(p1,p2+4)+0.009027777777777778*bcoef(p1+1,p2+4)+0.02291666666666667*bcoef(p1+2,p2+4)+0.009027777777777778*bcoef(p1+3,p2+4)+0.0003472222222222222*bcoef(p1+4,p2+4)-0.009027777777777778*bcoef(p1+3,p2)-0.0003472222222222222*bcoef(p1+4,p2)-0.003472222222222222*bcoef(p1,p2+1)-0.09027777777777778*bcoef(p1+1,p2+1)-0.2291666666666667*bcoef(p1+2,p2+1)-0.09027777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(1,1) = 0.001736111111111111*bcoef(p1,p2)+0.01736111111111111*bcoef(p1+1,p2)-0.01736111111111111*bcoef(p1+4,p2+1)-0.01736111111111111*bcoef(p1,p2+3)-0.1736111111111111*bcoef(p1+1,p2+3)+0.1736111111111111*bcoef(p1+3,p2+3)+0.01736111111111111*bcoef(p1+4,p2+3)-0.001736111111111111*bcoef(p1,p2+4)-0.01736111111111111*bcoef(p1+1,p2+4)+0.01736111111111111*bcoef(p1+3,p2+4)+0.001736111111111111*bcoef(p1+4,p2+4)-0.01736111111111111*bcoef(p1+3,p2)-0.001736111111111111*bcoef(p1+4,p2)+0.01736111111111111*bcoef(p1,p2+1)+0.1736111111111111*bcoef(p1+1,p2+1)-0.1736111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(2,1) = 0.03472222222222222*bcoef(p1,p2+3)-0.006944444444444444*bcoef(p1+1,p2)-0.03472222222222222*bcoef(p1+4,p2+1)-0.003472222222222222*bcoef(p1,p2)+0.06944444444444444*bcoef(p1+1,p2+3)+0.02083333333333333*bcoef(p1+2,p2)-0.2083333333333333*bcoef(p1+2,p2+3)+0.06944444444444444*bcoef(p1+3,p2+3)+0.03472222222222222*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1,p2+4)+0.006944444444444444*bcoef(p1+1,p2+4)-0.02083333333333333*bcoef(p1+2,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)-0.006944444444444444*bcoef(p1+3,p2)-0.003472222222222222*bcoef(p1+4,p2)-0.03472222222222222*bcoef(p1,p2+1)-0.06944444444444444*bcoef(p1+1,p2+1)+0.2083333333333333*bcoef(p1+2,p2+1)-0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(3,1) = 0.003472222222222222*bcoef(p1,p2)-0.006944444444444444*bcoef(p1+1,p2)-0.03472222222222222*bcoef(p1+4,p2+1)-0.03472222222222222*bcoef(p1,p2+3)+0.06944444444444444*bcoef(p1+1,p2+3)-0.06944444444444444*bcoef(p1+3,p2+3)+0.03472222222222222*bcoef(p1+4,p2+3)-0.003472222222222222*bcoef(p1,p2+4)+0.006944444444444444*bcoef(p1+1,p2+4)-0.006944444444444444*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)+0.006944444444444444*bcoef(p1+3,p2)-0.003472222222222222*bcoef(p1+4,p2)+0.03472222222222222*bcoef(p1,p2+1)-0.06944444444444444*bcoef(p1+1,p2+1)+0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(4,1) = 0.006944444444444444*bcoef(p1+1,p2)-0.001736111111111111*bcoef(p1,p2)-0.01736111111111111*bcoef(p1+4,p2+1)+0.01736111111111111*bcoef(p1,p2+3)-0.06944444444444444*bcoef(p1+1,p2+3)-0.01041666666666667*bcoef(p1+2,p2)+0.1041666666666667*bcoef(p1+2,p2+3)-0.06944444444444444*bcoef(p1+3,p2+3)+0.01736111111111111*bcoef(p1+4,p2+3)+0.001736111111111111*bcoef(p1,p2+4)-0.006944444444444444*bcoef(p1+1,p2+4)+0.01041666666666667*bcoef(p1+2,p2+4)-0.006944444444444444*bcoef(p1+3,p2+4)+0.001736111111111111*bcoef(p1+4,p2+4)+0.006944444444444444*bcoef(p1+3,p2)-0.001736111111111111*bcoef(p1+4,p2)-0.01736111111111111*bcoef(p1,p2+1)+0.06944444444444444*bcoef(p1+1,p2+1)-0.1041666666666667*bcoef(p1+2,p2+1)+0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(5,1) = 0.0003472222222222222*bcoef(p1,p2)-0.001736111111111111*bcoef(p1+1,p2)+0.01736111111111111*bcoef(p1+4,p2+1)-0.003472222222222222*bcoef(p1+5,p2+1)-0.003472222222222222*bcoef(p1,p2+3)+0.01736111111111111*bcoef(p1+1,p2+3)+0.003472222222222222*bcoef(p1+2,p2)-0.03472222222222222*bcoef(p1+2,p2+3)+0.03472222222222222*bcoef(p1+3,p2+3)-0.01736111111111111*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1+5,p2+3)-0.0003472222222222222*bcoef(p1,p2+4)+0.001736111111111111*bcoef(p1+1,p2+4)-0.003472222222222222*bcoef(p1+2,p2+4)+0.003472222222222222*bcoef(p1+3,p2+4)-0.001736111111111111*bcoef(p1+4,p2+4)+0.0003472222222222222*bcoef(p1+5,p2+4)-0.003472222222222222*bcoef(p1+3,p2)+0.001736111111111111*bcoef(p1+4,p2)-0.0003472222222222222*bcoef(p1+5,p2)+0.003472222222222222*bcoef(p1,p2+1)-0.01736111111111111*bcoef(p1+1,p2+1)+0.03472222222222222*bcoef(p1+2,p2+1)-0.03472222222222222*bcoef(p1+3,p2+1);
        coef_mat_buf(0,2) = 0.0006944444444444444*bcoef(p1,p2)+0.01805555555555556*bcoef(p1+1,p2)+0.001388888888888889*bcoef(p1+4,p2+1)-0.004166666666666667*bcoef(p1,p2+2)-0.1083333333333333*bcoef(p1+1,p2+2)-0.275*bcoef(p1+2,p2+2)-0.1083333333333333*bcoef(p1+3,p2+2)-0.004166666666666667*bcoef(p1+4,p2+2)+0.001388888888888889*bcoef(p1,p2+3)+0.03611111111111111*bcoef(p1+1,p2+3)+0.04583333333333333*bcoef(p1+2,p2)+0.09166666666666667*bcoef(p1+2,p2+3)+0.03611111111111111*bcoef(p1+3,p2+3)+0.001388888888888889*bcoef(p1+4,p2+3)+0.0006944444444444444*bcoef(p1,p2+4)+0.01805555555555556*bcoef(p1+1,p2+4)+0.04583333333333333*bcoef(p1+2,p2+4)+0.01805555555555556*bcoef(p1+3,p2+4)+0.0006944444444444444*bcoef(p1+4,p2+4)+0.01805555555555556*bcoef(p1+3,p2)+0.0006944444444444444*bcoef(p1+4,p2)+0.001388888888888889*bcoef(p1,p2+1)+0.03611111111111111*bcoef(p1+1,p2+1)+0.09166666666666667*bcoef(p1+2,p2+1)+0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(1,2) = 0.006944444444444444*bcoef(p1+4,p2+1)-0.03472222222222222*bcoef(p1+1,p2)-0.003472222222222222*bcoef(p1,p2)+0.02083333333333333*bcoef(p1,p2+2)+0.2083333333333333*bcoef(p1+1,p2+2)-0.2083333333333333*bcoef(p1+3,p2+2)-0.02083333333333333*bcoef(p1+4,p2+2)-0.006944444444444444*bcoef(p1,p2+3)-0.06944444444444444*bcoef(p1+1,p2+3)+0.06944444444444444*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)-0.003472222222222222*bcoef(p1,p2+4)-0.03472222222222222*bcoef(p1+1,p2+4)+0.03472222222222222*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)+0.03472222222222222*bcoef(p1+3,p2)+0.003472222222222222*bcoef(p1+4,p2)-0.006944444444444444*bcoef(p1,p2+1)-0.06944444444444444*bcoef(p1+1,p2+1)+0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(2,2) = 0.006944444444444444*bcoef(p1,p2)+0.01388888888888889*bcoef(p1+1,p2)+0.01388888888888889*bcoef(p1+4,p2+1)-0.04166666666666667*bcoef(p1,p2+2)-0.08333333333333333*bcoef(p1+1,p2+2)+0.25*bcoef(p1+2,p2+2)-0.08333333333333333*bcoef(p1+3,p2+2)-0.04166666666666667*bcoef(p1+4,p2+2)+0.01388888888888889*bcoef(p1,p2+3)+0.02777777777777778*bcoef(p1+1,p2+3)-0.04166666666666667*bcoef(p1+2,p2)-0.08333333333333333*bcoef(p1+2,p2+3)+0.02777777777777778*bcoef(p1+3,p2+3)+0.01388888888888889*bcoef(p1+4,p2+3)+0.006944444444444444*bcoef(p1,p2+4)+0.01388888888888889*bcoef(p1+1,p2+4)-0.04166666666666667*bcoef(p1+2,p2+4)+0.01388888888888889*bcoef(p1+3,p2+4)+0.006944444444444444*bcoef(p1+4,p2+4)+0.01388888888888889*bcoef(p1+3,p2)+0.006944444444444444*bcoef(p1+4,p2)+0.01388888888888889*bcoef(p1,p2+1)+0.02777777777777778*bcoef(p1+1,p2+1)-0.08333333333333333*bcoef(p1+2,p2+1)+0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(3,2) = 0.01388888888888889*bcoef(p1+1,p2)-0.006944444444444444*bcoef(p1,p2)+0.01388888888888889*bcoef(p1+4,p2+1)+0.04166666666666667*bcoef(p1,p2+2)-0.08333333333333333*bcoef(p1+1,p2+2)+0.08333333333333333*bcoef(p1+3,p2+2)-0.04166666666666667*bcoef(p1+4,p2+2)-0.01388888888888889*bcoef(p1,p2+3)+0.02777777777777778*bcoef(p1+1,p2+3)-0.02777777777777778*bcoef(p1+3,p2+3)+0.01388888888888889*bcoef(p1+4,p2+3)-0.006944444444444444*bcoef(p1,p2+4)+0.01388888888888889*bcoef(p1+1,p2+4)-0.01388888888888889*bcoef(p1+3,p2+4)+0.006944444444444444*bcoef(p1+4,p2+4)-0.01388888888888889*bcoef(p1+3,p2)+0.006944444444444444*bcoef(p1+4,p2)-0.01388888888888889*bcoef(p1,p2+1)+0.02777777777777778*bcoef(p1+1,p2+1)-0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(4,2) = 0.003472222222222222*bcoef(p1,p2)-0.01388888888888889*bcoef(p1+1,p2)+0.006944444444444444*bcoef(p1+4,p2+1)-0.02083333333333333*bcoef(p1,p2+2)+0.08333333333333333*bcoef(p1+1,p2+2)-0.125*bcoef(p1+2,p2+2)+0.08333333333333333*bcoef(p1+3,p2+2)-0.02083333333333333*bcoef(p1+4,p2+2)+0.006944444444444444*bcoef(p1,p2+3)-0.02777777777777778*bcoef(p1+1,p2+3)+0.02083333333333333*bcoef(p1+2,p2)+0.04166666666666667*bcoef(p1+2,p2+3)-0.02777777777777778*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1,p2+4)-0.01388888888888889*bcoef(p1+1,p2+4)+0.02083333333333333*bcoef(p1+2,p2+4)-0.01388888888888889*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)-0.01388888888888889*bcoef(p1+3,p2)+0.003472222222222222*bcoef(p1+4,p2)+0.006944444444444444*bcoef(p1,p2+1)-0.02777777777777778*bcoef(p1+1,p2+1)+0.04166666666666667*bcoef(p1+2,p2+1)-0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(5,2) = 0.003472222222222222*bcoef(p1+1,p2)-0.0006944444444444444*bcoef(p1,p2)-0.006944444444444444*bcoef(p1+4,p2+1)+0.001388888888888889*bcoef(p1+5,p2+1)+0.004166666666666667*bcoef(p1,p2+2)-0.02083333333333333*bcoef(p1+1,p2+2)+0.04166666666666667*bcoef(p1+2,p2+2)-0.04166666666666667*bcoef(p1+3,p2+2)+0.02083333333333333*bcoef(p1+4,p2+2)-0.004166666666666667*bcoef(p1+5,p2+2)-0.001388888888888889*bcoef(p1,p2+3)+0.006944444444444444*bcoef(p1+1,p2+3)-0.006944444444444444*bcoef(p1+2,p2)-0.01388888888888889*bcoef(p1+2,p2+3)+0.01388888888888889*bcoef(p1+3,p2+3)-0.006944444444444444*bcoef(p1+4,p2+3)+0.001388888888888889*bcoef(p1+5,p2+3)-0.0006944444444444444*bcoef(p1,p2+4)+0.003472222222222222*bcoef(p1+1,p2+4)-0.006944444444444444*bcoef(p1+2,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)-0.003472222222222222*bcoef(p1+4,p2+4)+0.0006944444444444444*bcoef(p1+5,p2+4)+0.006944444444444444*bcoef(p1+3,p2)-0.003472222222222222*bcoef(p1+4,p2)+0.0006944444444444444*bcoef(p1+5,p2)-0.001388888888888889*bcoef(p1,p2+1)+0.006944444444444444*bcoef(p1+1,p2+1)-0.01388888888888889*bcoef(p1+2,p2+1)+0.01388888888888889*bcoef(p1+3,p2+1);
        coef_mat_buf(0,3) = 0.001388888888888889*bcoef(p1+4,p2+1)-0.01805555555555556*bcoef(p1+1,p2)-0.0006944444444444444*bcoef(p1,p2)-0.001388888888888889*bcoef(p1,p2+3)-0.03611111111111111*bcoef(p1+1,p2+3)-0.04583333333333333*bcoef(p1+2,p2)-0.09166666666666667*bcoef(p1+2,p2+3)-0.03611111111111111*bcoef(p1+3,p2+3)-0.001388888888888889*bcoef(p1+4,p2+3)+0.0006944444444444444*bcoef(p1,p2+4)+0.01805555555555556*bcoef(p1+1,p2+4)+0.04583333333333333*bcoef(p1+2,p2+4)+0.01805555555555556*bcoef(p1+3,p2+4)+0.0006944444444444444*bcoef(p1+4,p2+4)-0.01805555555555556*bcoef(p1+3,p2)-0.0006944444444444444*bcoef(p1+4,p2)+0.001388888888888889*bcoef(p1,p2+1)+0.03611111111111111*bcoef(p1+1,p2+1)+0.09166666666666667*bcoef(p1+2,p2+1)+0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(1,3) = 0.003472222222222222*bcoef(p1,p2)+0.03472222222222222*bcoef(p1+1,p2)+0.006944444444444444*bcoef(p1+4,p2+1)+0.006944444444444444*bcoef(p1,p2+3)+0.06944444444444444*bcoef(p1+1,p2+3)-0.06944444444444444*bcoef(p1+3,p2+3)-0.006944444444444444*bcoef(p1+4,p2+3)-0.003472222222222222*bcoef(p1,p2+4)-0.03472222222222222*bcoef(p1+1,p2+4)+0.03472222222222222*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)-0.03472222222222222*bcoef(p1+3,p2)-0.003472222222222222*bcoef(p1+4,p2)-0.006944444444444444*bcoef(p1,p2+1)-0.06944444444444444*bcoef(p1+1,p2+1)+0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(2,3) = 0.01388888888888889*bcoef(p1+4,p2+1)-0.01388888888888889*bcoef(p1+1,p2)-0.006944444444444444*bcoef(p1,p2)-0.01388888888888889*bcoef(p1,p2+3)-0.02777777777777778*bcoef(p1+1,p2+3)+0.04166666666666667*bcoef(p1+2,p2)+0.08333333333333333*bcoef(p1+2,p2+3)-0.02777777777777778*bcoef(p1+3,p2+3)-0.01388888888888889*bcoef(p1+4,p2+3)+0.006944444444444444*bcoef(p1,p2+4)+0.01388888888888889*bcoef(p1+1,p2+4)-0.04166666666666667*bcoef(p1+2,p2+4)+0.01388888888888889*bcoef(p1+3,p2+4)+0.006944444444444444*bcoef(p1+4,p2+4)-0.01388888888888889*bcoef(p1+3,p2)-0.006944444444444444*bcoef(p1+4,p2)+0.01388888888888889*bcoef(p1,p2+1)+0.02777777777777778*bcoef(p1+1,p2+1)-0.08333333333333333*bcoef(p1+2,p2+1)+0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(3,3) = 0.006944444444444444*bcoef(p1,p2)-0.01388888888888889*bcoef(p1+1,p2)+0.01388888888888889*bcoef(p1+4,p2+1)+0.01388888888888889*bcoef(p1,p2+3)-0.02777777777777778*bcoef(p1+1,p2+3)+0.02777777777777778*bcoef(p1+3,p2+3)-0.01388888888888889*bcoef(p1+4,p2+3)-0.006944444444444444*bcoef(p1,p2+4)+0.01388888888888889*bcoef(p1+1,p2+4)-0.01388888888888889*bcoef(p1+3,p2+4)+0.006944444444444444*bcoef(p1+4,p2+4)+0.01388888888888889*bcoef(p1+3,p2)-0.006944444444444444*bcoef(p1+4,p2)-0.01388888888888889*bcoef(p1,p2+1)+0.02777777777777778*bcoef(p1+1,p2+1)-0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(4,3) = 0.01388888888888889*bcoef(p1+1,p2)-0.003472222222222222*bcoef(p1,p2)+0.006944444444444444*bcoef(p1+4,p2+1)-0.006944444444444444*bcoef(p1,p2+3)+0.02777777777777778*bcoef(p1+1,p2+3)-0.02083333333333333*bcoef(p1+2,p2)-0.04166666666666667*bcoef(p1+2,p2+3)+0.02777777777777778*bcoef(p1+3,p2+3)-0.006944444444444444*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1,p2+4)-0.01388888888888889*bcoef(p1+1,p2+4)+0.02083333333333333*bcoef(p1+2,p2+4)-0.01388888888888889*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)+0.01388888888888889*bcoef(p1+3,p2)-0.003472222222222222*bcoef(p1+4,p2)+0.006944444444444444*bcoef(p1,p2+1)-0.02777777777777778*bcoef(p1+1,p2+1)+0.04166666666666667*bcoef(p1+2,p2+1)-0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(5,3) = 0.0006944444444444444*bcoef(p1,p2)-0.003472222222222222*bcoef(p1+1,p2)-0.006944444444444444*bcoef(p1+4,p2+1)+0.001388888888888889*bcoef(p1+5,p2+1)+0.001388888888888889*bcoef(p1,p2+3)-0.006944444444444444*bcoef(p1+1,p2+3)+0.006944444444444444*bcoef(p1+2,p2)+0.01388888888888889*bcoef(p1+2,p2+3)-0.01388888888888889*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)-0.001388888888888889*bcoef(p1+5,p2+3)-0.0006944444444444444*bcoef(p1,p2+4)+0.003472222222222222*bcoef(p1+1,p2+4)-0.006944444444444444*bcoef(p1+2,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)-0.003472222222222222*bcoef(p1+4,p2+4)+0.0006944444444444444*bcoef(p1+5,p2+4)-0.006944444444444444*bcoef(p1+3,p2)+0.003472222222222222*bcoef(p1+4,p2)-0.0006944444444444444*bcoef(p1+5,p2)-0.001388888888888889*bcoef(p1,p2+1)+0.006944444444444444*bcoef(p1+1,p2+1)-0.01388888888888889*bcoef(p1+2,p2+1)+0.01388888888888889*bcoef(p1+3,p2+1);
        coef_mat_buf(0,4) = 0.0003472222222222222*bcoef(p1,p2)+0.009027777777777778*bcoef(p1+1,p2)-0.001388888888888889*bcoef(p1+4,p2+1)+0.002083333333333333*bcoef(p1,p2+2)+0.05416666666666667*bcoef(p1+1,p2+2)+0.1375*bcoef(p1+2,p2+2)+0.05416666666666667*bcoef(p1+3,p2+2)+0.002083333333333333*bcoef(p1+4,p2+2)-0.001388888888888889*bcoef(p1,p2+3)-0.03611111111111111*bcoef(p1+1,p2+3)+0.02291666666666667*bcoef(p1+2,p2)-0.09166666666666667*bcoef(p1+2,p2+3)-0.03611111111111111*bcoef(p1+3,p2+3)-0.001388888888888889*bcoef(p1+4,p2+3)+0.0003472222222222222*bcoef(p1,p2+4)+0.009027777777777778*bcoef(p1+1,p2+4)+0.02291666666666667*bcoef(p1+2,p2+4)+0.009027777777777778*bcoef(p1+3,p2+4)+0.0003472222222222222*bcoef(p1+4,p2+4)+0.009027777777777778*bcoef(p1+3,p2)+0.0003472222222222222*bcoef(p1+4,p2)-0.001388888888888889*bcoef(p1,p2+1)-0.03611111111111111*bcoef(p1+1,p2+1)-0.09166666666666667*bcoef(p1+2,p2+1)-0.03611111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(1,4) = 0.1041666666666667*bcoef(p1+3,p2+2)-0.01736111111111111*bcoef(p1+1,p2)-0.006944444444444444*bcoef(p1+4,p2+1)-0.01041666666666667*bcoef(p1,p2+2)-0.1041666666666667*bcoef(p1+1,p2+2)-0.001736111111111111*bcoef(p1,p2)+0.01041666666666667*bcoef(p1+4,p2+2)+0.006944444444444444*bcoef(p1,p2+3)+0.06944444444444444*bcoef(p1+1,p2+3)-0.06944444444444444*bcoef(p1+3,p2+3)-0.006944444444444444*bcoef(p1+4,p2+3)-0.001736111111111111*bcoef(p1,p2+4)-0.01736111111111111*bcoef(p1+1,p2+4)+0.01736111111111111*bcoef(p1+3,p2+4)+0.001736111111111111*bcoef(p1+4,p2+4)+0.01736111111111111*bcoef(p1+3,p2)+0.001736111111111111*bcoef(p1+4,p2)+0.006944444444444444*bcoef(p1,p2+1)+0.06944444444444444*bcoef(p1+1,p2+1)-0.06944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(2,4) = 0.003472222222222222*bcoef(p1,p2)+0.006944444444444444*bcoef(p1+1,p2)-0.01388888888888889*bcoef(p1+4,p2+1)+0.02083333333333333*bcoef(p1,p2+2)+0.04166666666666667*bcoef(p1+1,p2+2)-0.125*bcoef(p1+2,p2+2)+0.04166666666666667*bcoef(p1+3,p2+2)+0.02083333333333333*bcoef(p1+4,p2+2)-0.01388888888888889*bcoef(p1,p2+3)-0.02777777777777778*bcoef(p1+1,p2+3)-0.02083333333333333*bcoef(p1+2,p2)+0.08333333333333333*bcoef(p1+2,p2+3)-0.02777777777777778*bcoef(p1+3,p2+3)-0.01388888888888889*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1,p2+4)+0.006944444444444444*bcoef(p1+1,p2+4)-0.02083333333333333*bcoef(p1+2,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)+0.006944444444444444*bcoef(p1+3,p2)+0.003472222222222222*bcoef(p1+4,p2)-0.01388888888888889*bcoef(p1,p2+1)-0.02777777777777778*bcoef(p1+1,p2+1)+0.08333333333333333*bcoef(p1+2,p2+1)-0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(3,4) = 0.006944444444444444*bcoef(p1+1,p2)-0.003472222222222222*bcoef(p1,p2)-0.01388888888888889*bcoef(p1+4,p2+1)-0.02083333333333333*bcoef(p1,p2+2)+0.04166666666666667*bcoef(p1+1,p2+2)-0.04166666666666667*bcoef(p1+3,p2+2)+0.02083333333333333*bcoef(p1+4,p2+2)+0.01388888888888889*bcoef(p1,p2+3)-0.02777777777777778*bcoef(p1+1,p2+3)+0.02777777777777778*bcoef(p1+3,p2+3)-0.01388888888888889*bcoef(p1+4,p2+3)-0.003472222222222222*bcoef(p1,p2+4)+0.006944444444444444*bcoef(p1+1,p2+4)-0.006944444444444444*bcoef(p1+3,p2+4)+0.003472222222222222*bcoef(p1+4,p2+4)-0.006944444444444444*bcoef(p1+3,p2)+0.003472222222222222*bcoef(p1+4,p2)+0.01388888888888889*bcoef(p1,p2+1)-0.02777777777777778*bcoef(p1+1,p2+1)+0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(4,4) = 0.001736111111111111*bcoef(p1,p2)-0.006944444444444444*bcoef(p1+1,p2)-0.006944444444444444*bcoef(p1+4,p2+1)+0.01041666666666667*bcoef(p1,p2+2)-0.04166666666666667*bcoef(p1+1,p2+2)+0.0625*bcoef(p1+2,p2+2)-0.04166666666666667*bcoef(p1+3,p2+2)+0.01041666666666667*bcoef(p1+4,p2+2)-0.006944444444444444*bcoef(p1,p2+3)+0.02777777777777778*bcoef(p1+1,p2+3)+0.01041666666666667*bcoef(p1+2,p2)-0.04166666666666667*bcoef(p1+2,p2+3)+0.02777777777777778*bcoef(p1+3,p2+3)-0.006944444444444444*bcoef(p1+4,p2+3)+0.001736111111111111*bcoef(p1,p2+4)-0.006944444444444444*bcoef(p1+1,p2+4)+0.01041666666666667*bcoef(p1+2,p2+4)-0.006944444444444444*bcoef(p1+3,p2+4)+0.001736111111111111*bcoef(p1+4,p2+4)-0.006944444444444444*bcoef(p1+3,p2)+0.001736111111111111*bcoef(p1+4,p2)-0.006944444444444444*bcoef(p1,p2+1)+0.02777777777777778*bcoef(p1+1,p2+1)-0.04166666666666667*bcoef(p1+2,p2+1)+0.02777777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(5,4) = 0.001736111111111111*bcoef(p1+1,p2)-0.0003472222222222222*bcoef(p1,p2)+0.006944444444444444*bcoef(p1+4,p2+1)-0.001388888888888889*bcoef(p1+5,p2+1)-0.002083333333333333*bcoef(p1,p2+2)+0.01041666666666667*bcoef(p1+1,p2+2)-0.02083333333333333*bcoef(p1+2,p2+2)+0.02083333333333333*bcoef(p1+3,p2+2)-0.01041666666666667*bcoef(p1+4,p2+2)+0.002083333333333333*bcoef(p1+5,p2+2)+0.001388888888888889*bcoef(p1,p2+3)-0.006944444444444444*bcoef(p1+1,p2+3)-0.003472222222222222*bcoef(p1+2,p2)+0.01388888888888889*bcoef(p1+2,p2+3)-0.01388888888888889*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)-0.001388888888888889*bcoef(p1+5,p2+3)-0.0003472222222222222*bcoef(p1,p2+4)+0.001736111111111111*bcoef(p1+1,p2+4)-0.003472222222222222*bcoef(p1+2,p2+4)+0.003472222222222222*bcoef(p1+3,p2+4)-0.001736111111111111*bcoef(p1+4,p2+4)+0.0003472222222222222*bcoef(p1+5,p2+4)+0.003472222222222222*bcoef(p1+3,p2)-0.001736111111111111*bcoef(p1+4,p2)+0.0003472222222222222*bcoef(p1+5,p2)+0.001388888888888889*bcoef(p1,p2+1)-0.006944444444444444*bcoef(p1+1,p2+1)+0.01388888888888889*bcoef(p1+2,p2+1)-0.01388888888888889*bcoef(p1+3,p2+1);
        coef_mat_buf(0,5) = 0.0003472222222222222*bcoef(p1+4,p2+1)-0.001805555555555556*bcoef(p1+1,p2)-0.00006944444444444444*bcoef(p1,p2)-0.0006944444444444444*bcoef(p1,p2+2)-0.01805555555555556*bcoef(p1+1,p2+2)-0.04583333333333333*bcoef(p1+2,p2+2)-0.01805555555555556*bcoef(p1+3,p2+2)-0.0006944444444444444*bcoef(p1+4,p2+2)+0.0006944444444444444*bcoef(p1,p2+3)+0.01805555555555556*bcoef(p1+1,p2+3)-0.004583333333333333*bcoef(p1+2,p2)+0.04583333333333333*bcoef(p1+2,p2+3)+0.01805555555555556*bcoef(p1+3,p2+3)+0.0006944444444444444*bcoef(p1+4,p2+3)-0.0003472222222222222*bcoef(p1,p2+4)-0.009027777777777778*bcoef(p1+1,p2+4)-0.02291666666666667*bcoef(p1+2,p2+4)-0.009027777777777778*bcoef(p1+3,p2+4)-0.0003472222222222222*bcoef(p1+4,p2+4)-0.001805555555555556*bcoef(p1+3,p2)+0.00006944444444444444*bcoef(p1,p2+5)+0.001805555555555556*bcoef(p1+1,p2+5)+0.004583333333333333*bcoef(p1+2,p2+5)+0.001805555555555556*bcoef(p1+3,p2+5)+0.00006944444444444444*bcoef(p1+4,p2+5)-0.00006944444444444444*bcoef(p1+4,p2)+0.0003472222222222222*bcoef(p1,p2+1)+0.009027777777777778*bcoef(p1+1,p2+1)+0.02291666666666667*bcoef(p1+2,p2+1)+0.009027777777777778*bcoef(p1+3,p2+1);
        coef_mat_buf(1,5) = 0.0003472222222222222*bcoef(p1,p2)+0.003472222222222222*bcoef(p1+1,p2)+0.001736111111111111*bcoef(p1+4,p2+1)+0.003472222222222222*bcoef(p1,p2+2)+0.03472222222222222*bcoef(p1+1,p2+2)-0.03472222222222222*bcoef(p1+3,p2+2)-0.003472222222222222*bcoef(p1+4,p2+2)-0.003472222222222222*bcoef(p1,p2+3)-0.03472222222222222*bcoef(p1+1,p2+3)+0.03472222222222222*bcoef(p1+3,p2+3)+0.003472222222222222*bcoef(p1+4,p2+3)+0.001736111111111111*bcoef(p1,p2+4)+0.01736111111111111*bcoef(p1+1,p2+4)-0.01736111111111111*bcoef(p1+3,p2+4)-0.001736111111111111*bcoef(p1+4,p2+4)-0.003472222222222222*bcoef(p1+3,p2)-0.0003472222222222222*bcoef(p1,p2+5)-0.003472222222222222*bcoef(p1+1,p2+5)+0.003472222222222222*bcoef(p1+3,p2+5)+0.0003472222222222222*bcoef(p1+4,p2+5)-0.0003472222222222222*bcoef(p1+4,p2)-0.001736111111111111*bcoef(p1,p2+1)-0.01736111111111111*bcoef(p1+1,p2+1)+0.01736111111111111*bcoef(p1+3,p2+1);
        coef_mat_buf(2,5) = 0.003472222222222222*bcoef(p1+4,p2+1)-0.001388888888888889*bcoef(p1+1,p2)-0.0006944444444444444*bcoef(p1,p2)-0.006944444444444444*bcoef(p1,p2+2)-0.01388888888888889*bcoef(p1+1,p2+2)+0.04166666666666667*bcoef(p1+2,p2+2)-0.01388888888888889*bcoef(p1+3,p2+2)-0.006944444444444444*bcoef(p1+4,p2+2)+0.006944444444444444*bcoef(p1,p2+3)+0.01388888888888889*bcoef(p1+1,p2+3)+0.004166666666666667*bcoef(p1+2,p2)-0.04166666666666667*bcoef(p1+2,p2+3)+0.01388888888888889*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)-0.003472222222222222*bcoef(p1,p2+4)-0.006944444444444444*bcoef(p1+1,p2+4)+0.02083333333333333*bcoef(p1+2,p2+4)-0.006944444444444444*bcoef(p1+3,p2+4)-0.003472222222222222*bcoef(p1+4,p2+4)-0.001388888888888889*bcoef(p1+3,p2)+0.0006944444444444444*bcoef(p1,p2+5)+0.001388888888888889*bcoef(p1+1,p2+5)-0.004166666666666667*bcoef(p1+2,p2+5)+0.001388888888888889*bcoef(p1+3,p2+5)+0.0006944444444444444*bcoef(p1+4,p2+5)-0.0006944444444444444*bcoef(p1+4,p2)+0.003472222222222222*bcoef(p1,p2+1)+0.006944444444444444*bcoef(p1+1,p2+1)-0.02083333333333333*bcoef(p1+2,p2+1)+0.006944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(3,5) = 0.0006944444444444444*bcoef(p1,p2)-0.001388888888888889*bcoef(p1+1,p2)+0.003472222222222222*bcoef(p1+4,p2+1)+0.006944444444444444*bcoef(p1,p2+2)-0.01388888888888889*bcoef(p1+1,p2+2)+0.01388888888888889*bcoef(p1+3,p2+2)-0.006944444444444444*bcoef(p1+4,p2+2)-0.006944444444444444*bcoef(p1,p2+3)+0.01388888888888889*bcoef(p1+1,p2+3)-0.01388888888888889*bcoef(p1+3,p2+3)+0.006944444444444444*bcoef(p1+4,p2+3)+0.003472222222222222*bcoef(p1,p2+4)-0.006944444444444444*bcoef(p1+1,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)-0.003472222222222222*bcoef(p1+4,p2+4)+0.001388888888888889*bcoef(p1+3,p2)-0.0006944444444444444*bcoef(p1,p2+5)+0.001388888888888889*bcoef(p1+1,p2+5)-0.001388888888888889*bcoef(p1+3,p2+5)+0.0006944444444444444*bcoef(p1+4,p2+5)-0.0006944444444444444*bcoef(p1+4,p2)-0.003472222222222222*bcoef(p1,p2+1)+0.006944444444444444*bcoef(p1+1,p2+1)-0.006944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(4,5) = 0.001388888888888889*bcoef(p1+1,p2)-0.0003472222222222222*bcoef(p1,p2)+0.001736111111111111*bcoef(p1+4,p2+1)-0.003472222222222222*bcoef(p1,p2+2)+0.01388888888888889*bcoef(p1+1,p2+2)-0.02083333333333333*bcoef(p1+2,p2+2)+0.01388888888888889*bcoef(p1+3,p2+2)-0.003472222222222222*bcoef(p1+4,p2+2)+0.003472222222222222*bcoef(p1,p2+3)-0.01388888888888889*bcoef(p1+1,p2+3)-0.002083333333333333*bcoef(p1+2,p2)+0.02083333333333333*bcoef(p1+2,p2+3)-0.01388888888888889*bcoef(p1+3,p2+3)+0.003472222222222222*bcoef(p1+4,p2+3)-0.001736111111111111*bcoef(p1,p2+4)+0.006944444444444444*bcoef(p1+1,p2+4)-0.01041666666666667*bcoef(p1+2,p2+4)+0.006944444444444444*bcoef(p1+3,p2+4)-0.001736111111111111*bcoef(p1+4,p2+4)+0.001388888888888889*bcoef(p1+3,p2)+0.0003472222222222222*bcoef(p1,p2+5)-0.001388888888888889*bcoef(p1+1,p2+5)+0.002083333333333333*bcoef(p1+2,p2+5)-0.001388888888888889*bcoef(p1+3,p2+5)+0.0003472222222222222*bcoef(p1+4,p2+5)-0.0003472222222222222*bcoef(p1+4,p2)+0.001736111111111111*bcoef(p1,p2+1)-0.006944444444444444*bcoef(p1+1,p2+1)+0.01041666666666667*bcoef(p1+2,p2+1)-0.006944444444444444*bcoef(p1+3,p2+1);
        coef_mat_buf(5,5) = 0.00006944444444444444*bcoef(p1,p2)-0.0003472222222222222*bcoef(p1+1,p2)-0.001736111111111111*bcoef(p1+4,p2+1)+0.0003472222222222222*bcoef(p1+5,p2+1)+0.0006944444444444444*bcoef(p1,p2+2)-0.003472222222222222*bcoef(p1+1,p2+2)+0.006944444444444444*bcoef(p1+2,p2+2)-0.006944444444444444*bcoef(p1+3,p2+2)+0.003472222222222222*bcoef(p1+4,p2+2)-0.0006944444444444444*bcoef(p1+5,p2+2)-0.0006944444444444444*bcoef(p1,p2+3)+0.003472222222222222*bcoef(p1+1,p2+3)+0.0006944444444444444*bcoef(p1+2,p2)-0.006944444444444444*bcoef(p1+2,p2+3)+0.006944444444444444*bcoef(p1+3,p2+3)-0.003472222222222222*bcoef(p1+4,p2+3)+0.0006944444444444444*bcoef(p1+5,p2+3)+0.0003472222222222222*bcoef(p1,p2+4)-0.001736111111111111*bcoef(p1+1,p2+4)+0.003472222222222222*bcoef(p1+2,p2+4)-0.003472222222222222*bcoef(p1+3,p2+4)+0.001736111111111111*bcoef(p1+4,p2+4)-0.0003472222222222222*bcoef(p1+5,p2+4)-0.0006944444444444444*bcoef(p1+3,p2)-0.00006944444444444444*bcoef(p1,p2+5)+0.0003472222222222222*bcoef(p1+1,p2+5)-0.0006944444444444444*bcoef(p1+2,p2+5)+0.0006944444444444444*bcoef(p1+3,p2+5)-0.0003472222222222222*bcoef(p1+4,p2+5)+0.00006944444444444444*bcoef(p1+5,p2+5)+0.0003472222222222222*bcoef(p1+4,p2)-0.00006944444444444444*bcoef(p1+5,p2)-0.0003472222222222222*bcoef(p1,p2+1)+0.001736111111111111*bcoef(p1+1,p2+1)-0.003472222222222222*bcoef(p1+2,p2+1)+0.003472222222222222*bcoef(p1+3,p2+1);

        return coef_mat_buf;
    }
            
    // Precomputed Biquintic B-spline Interpolator ---------------------------//
    template <typename T_container> 
    quintic_interp_precompute<T_container>::quintic_interp_precompute(const_container &A) : quintic_interp_base<container>(A) { 
        // pre-allocate memory for entire coefficient array - note that this will 
        // be a lot of memory - 36 times the size of the original array.
        coef_mat_precompute_ptr = std::make_shared<Array2D<container>>(this->A_ptr->height(), this->A_ptr->width(), container(6,6));

        auto bcoef = this->get_bspline_mat_ptr(A);
        for (difference_type p2 = 0; p2 < this->A_ptr->width(); ++p2) {
            for (difference_type p1 = 0; p1 < this->A_ptr->height(); ++p1) {
                this->calc_coef_mat((*coef_mat_precompute_ptr)(p1,p2), *bcoef, p1 + this->bcoef_border - 2, p2 + this->bcoef_border - 2);
            }
        }
    } 
        
} // namespace details

// General Functions ---------------------------------------------------------//
template <typename T = double,typename T_alloc = std::allocator<T>>
Array2D<T,T_alloc> eye(typename Array2D<T,T_alloc>::difference_type n, T = T(), T_alloc = T_alloc()) {
    typedef typename Array2D<T,T_alloc>::difference_type        difference_type;
    
    if (n < 0) {
        throw std::invalid_argument("Attempted to create identity matrix with a size of: " + std::to_string(n) + ".");
    }
    
    Array2D<T,T_alloc> A(n,n);
    for (difference_type p = 0; p < n; ++p) {
        A(p,p) = 1;
    }
    
    return A;
}

} // namespace ncorr

#include "Array2DLinSolver.h"

#endif	/* ARRAY2D_H */
