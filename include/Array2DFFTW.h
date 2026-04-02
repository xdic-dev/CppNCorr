#ifndef ARRAY2D_FFTW_H
#define ARRAY2D_FFTW_H

#include "Array2D.h"

#include "fftw3.h"

namespace ncorr {

namespace details {
    extern std::mutex fftw_mutex;

    template <typename T>
    class fftw_allocator final {
        public:
            typedef T                                                value_type;
            typedef T*                                                  pointer;
            typedef const T*                                      const_pointer;
            typedef T&                                                reference;
            typedef const T&                                    const_reference;
            typedef std::size_t                                       size_type;
            typedef std::ptrdiff_t                              difference_type;

            fftw_allocator() noexcept = default;
            fftw_allocator(const fftw_allocator&) noexcept = default;
            fftw_allocator(fftw_allocator&&) noexcept = default;
            fftw_allocator& operator=(const fftw_allocator&) = default;
            fftw_allocator& operator=(fftw_allocator&&) = default;
            ~fftw_allocator() noexcept = default;

            pointer allocate(difference_type s, const void * = 0) {
                pointer ptr = static_cast<pointer>(malloc_function(s * sizeof(value_type)));
                if (!ptr) {
                    std::cerr << "Failed to allocate memory using allocate in fftw_allocator." << std::endl;
                    throw std::bad_alloc();
                }

                return ptr;
            }
            void deallocate(pointer ptr, difference_type) { free_function(ptr); }
            void construct(pointer ptr, const_reference val = value_type()) { ::new((void *)ptr) value_type(val); }
            void destroy(pointer ptr) { ptr->~value_type(); }
            template<typename T2>
            struct rebind { typedef fftw_allocator<T2> other; };

        private:
            void* malloc_function(difference_type s) { return malloc(s); }
            void free_function(pointer ptr) { return free(ptr); }
    };

    template <>
    inline void* fftw_allocator<double>::malloc_function(difference_type s) { return fftw_malloc(s); }

    template <>
    inline void fftw_allocator<double>::free_function(pointer ptr) { fftw_free(ptr); }
}

} // namespace ncorr

#endif
