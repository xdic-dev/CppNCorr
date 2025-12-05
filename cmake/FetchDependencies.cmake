# FetchDependencies.cmake
# Provides FetchContent fallbacks for dependencies when system packages are unavailable
# This is useful for servers without root access

include(FetchContent)

# Option to force FetchContent for all dependencies
option(FORCE_FETCH_DEPENDENCIES "Force using FetchContent for all dependencies" OFF)

#------------------------------------------------------------------------------
# nlohmann_json - Header-only JSON library (easiest to fetch)
#------------------------------------------------------------------------------
macro(find_or_fetch_nlohmann_json)
    if(NOT FORCE_FETCH_DEPENDENCIES)
        find_package(nlohmann_json CONFIG QUIET)
    endif()
    
    if(NOT nlohmann_json_FOUND)
        message(STATUS "nlohmann_json not found - fetching from GitHub...")
        FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG        v3.11.3
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(nlohmann_json)
        set(nlohmann_json_FOUND TRUE)
        message(STATUS "nlohmann_json fetched successfully")
    else()
        message(STATUS "Found system nlohmann_json")
    endif()
endmacro()

#------------------------------------------------------------------------------
# OpenCV - Computer vision library
# Note: OpenCV is complex to build from source. We provide a minimal fetch
# but recommend system installation when possible.
#------------------------------------------------------------------------------
macro(find_or_fetch_opencv)
    if(NOT FORCE_FETCH_DEPENDENCIES)
        find_package(OpenCV QUIET)
    endif()
    
    if(NOT OpenCV_FOUND)
        message(STATUS "OpenCV not found - fetching from GitHub...")
        message(STATUS "Note: Building OpenCV from source may take a long time")
        
        # Set OpenCV build options for minimal build
        set(BUILD_LIST "core,imgproc,imgcodecs,highgui,videoio" CACHE STRING "" FORCE)
        set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
        set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(WITH_CUDA OFF CACHE BOOL "" FORCE)
        set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
        set(WITH_VTK OFF CACHE BOOL "" FORCE)
        set(WITH_GTK OFF CACHE BOOL "" FORCE)
        set(WITH_QT OFF CACHE BOOL "" FORCE)
        set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
        set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
        set(WITH_V4L OFF CACHE BOOL "" FORCE)
        set(WITH_1394 OFF CACHE BOOL "" FORCE)
        set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
        set(WITH_JASPER OFF CACHE BOOL "" FORCE)
        set(WITH_WEBP OFF CACHE BOOL "" FORCE)
        set(WITH_TIFF OFF CACHE BOOL "" FORCE)
        set(WITH_PNG ON CACHE BOOL "" FORCE)
        set(WITH_JPEG ON CACHE BOOL "" FORCE)
        
        FetchContent_Declare(
            opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG        4.9.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(opencv)
        
        # Set variables that find_package would normally set
        set(OpenCV_FOUND TRUE)
        set(OpenCV_INCLUDE_DIRS ${opencv_SOURCE_DIR}/include ${opencv_BINARY_DIR})
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_videoio)
        
        message(STATUS "OpenCV fetched successfully")
    else()
        message(STATUS "Found system OpenCV: ${OpenCV_VERSION}")
    endif()
endmacro()

#------------------------------------------------------------------------------
# SuiteSparse - Sparse matrix library
# Note: SuiteSparse has complex dependencies (BLAS, LAPACK). 
# We use the official CMake-enabled repository.
#------------------------------------------------------------------------------
macro(find_or_fetch_suitesparse)
    if(NOT FORCE_FETCH_DEPENDENCIES)
        # Try to find SuiteSparse components
        find_library(SPQR_LIBRARY NAMES spqr QUIET)
        find_library(CHOLMOD_LIBRARY NAMES cholmod QUIET)
        find_library(SUITESPARSE_CONFIG_LIBRARY NAMES suitesparseconfig QUIET)
        find_library(AMD_LIBRARY NAMES amd QUIET)
        find_library(COLAMD_LIBRARY NAMES colamd QUIET)
    endif()
    
    if(NOT SPQR_LIBRARY OR NOT CHOLMOD_LIBRARY OR FORCE_FETCH_DEPENDENCIES)
        message(STATUS "SuiteSparse not found - fetching from GitHub...")
        
        # SuiteSparse build options
        set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;colamd;cholmod;spqr" CACHE STRING "" FORCE)
        set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "" FORCE)
        set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        
        FetchContent_Declare(
            suitesparse
            GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
            GIT_TAG        v7.6.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(suitesparse)
        
        set(SUITESPARSE_FOUND TRUE)
        set(SUITESPARSE_LIBRARIES 
            SuiteSparse::SPQR 
            SuiteSparse::CHOLMOD 
            SuiteSparse::SuiteSparseConfig 
            SuiteSparse::AMD 
            SuiteSparse::COLAMD
        )
        
        message(STATUS "SuiteSparse fetched successfully")
    else()
        message(STATUS "Found system SuiteSparse")
        set(SUITESPARSE_FOUND TRUE)
        set(SUITESPARSE_LIBRARIES 
            ${SPQR_LIBRARY} 
            ${CHOLMOD_LIBRARY} 
            ${SUITESPARSE_CONFIG_LIBRARY} 
            ${AMD_LIBRARY} 
            ${COLAMD_LIBRARY}
        )
    endif()
endmacro()

#------------------------------------------------------------------------------
# FFTW3 - Fast Fourier Transform library
#------------------------------------------------------------------------------
macro(find_or_fetch_fftw)
    if(NOT FORCE_FETCH_DEPENDENCIES)
        find_library(FFTW_LIBRARY NAMES fftw3 QUIET)
        find_path(FFTW_INCLUDE_DIR fftw3.h
            HINTS /opt/homebrew/include /usr/local/include /usr/include
        )
    endif()
    
    if(NOT FFTW_LIBRARY OR FORCE_FETCH_DEPENDENCIES)
        message(STATUS "FFTW not found - fetching from GitHub...")
        
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
        set(ENABLE_THREADS ON CACHE BOOL "" FORCE)
        
        FetchContent_Declare(
            fftw3
            URL https://www.fftw.org/fftw-3.3.10.tar.gz
            URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
        )
        FetchContent_MakeAvailable(fftw3)
        
        set(FFTW_FOUND TRUE)
        set(FFTW_LIBRARY fftw3)
        set(FFTW_INCLUDE_DIR ${fftw3_SOURCE_DIR}/api)
        
        message(STATUS "FFTW fetched successfully")
    else()
        message(STATUS "Found system FFTW: ${FFTW_LIBRARY}")
        set(FFTW_FOUND TRUE)
    endif()
endmacro()

#------------------------------------------------------------------------------
# OpenBLAS - BLAS/LAPACK implementation
# Used as fallback when system BLAS/LAPACK not available
#------------------------------------------------------------------------------
macro(find_or_fetch_openblas)
    if(NOT FORCE_FETCH_DEPENDENCIES)
        find_package(BLAS QUIET)
        find_package(LAPACK QUIET)
    endif()
    
    if(NOT BLAS_FOUND OR NOT LAPACK_FOUND OR FORCE_FETCH_DEPENDENCIES)
        message(STATUS "BLAS/LAPACK not found - fetching OpenBLAS from GitHub...")
        
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(USE_OPENMP OFF CACHE BOOL "" FORCE)
        set(NO_LAPACK OFF CACHE BOOL "" FORCE)
        
        FetchContent_Declare(
            openblas
            GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
            GIT_TAG        v0.3.26
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(openblas)
        
        set(BLAS_FOUND TRUE)
        set(LAPACK_FOUND TRUE)
        set(BLAS_LIBRARIES openblas)
        set(LAPACK_LIBRARIES openblas)
        
        message(STATUS "OpenBLAS fetched successfully (provides BLAS and LAPACK)")
    else()
        message(STATUS "Found system BLAS: ${BLAS_LIBRARIES}")
        message(STATUS "Found system LAPACK: ${LAPACK_LIBRARIES}")
    endif()
endmacro()

#------------------------------------------------------------------------------
# Convenience macro to fetch all dependencies
#------------------------------------------------------------------------------
macro(fetch_all_dependencies)
    find_or_fetch_nlohmann_json()
    find_or_fetch_fftw()
    find_or_fetch_openblas()
    find_or_fetch_suitesparse()
    find_or_fetch_opencv()
endmacro()
