CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(ncorr_library)

# Include FetchContent fallbacks for systems without root access
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDependencies)

# Option to force fetching all dependencies (useful for reproducible builds)
option(FORCE_FETCH_DEPENDENCIES "Force using FetchContent for all dependencies" OFF)

# Only tested for g++ on Ubuntu 12.04. This assumes all required libraries have been 
# installed, so directories to dependent libraries and their headers are not explicitly 
# included, since the install directories are searched automatically by g++.

# Set files
SET(ncorr_src src/ncorr.cpp src/Strain2D.cpp src/Disp2D.cpp src/Data2D.cpp src/ROI2D.cpp src/Image2D.cpp src/Array2D.cpp)
SET(ncorr_h include/ncorr.h include/Strain2D.h include/Disp2D.h include/Data2D.h include/ROI2D.h include/Image2D.h include/Array2D.h)

# Set include directory
INCLUDE_DIRECTORIES(include)

# Set output for library
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Add library
ADD_LIBRARY(ncorr STATIC ${ncorr_src})

# Set C++17 support (required for std::optional and modern features)
set_property(TARGET ncorr PROPERTY CXX_STANDARD 17)
set_property(TARGET ncorr PROPERTY CXX_STANDARD_REQUIRED ON)

# Set -03 optimization
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-O3" COMPILER_SUPPORTS_O3)
if (COMPILER_SUPPORTS_O3)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Find OpenMP - with special handling for Mac Homebrew
if(APPLE)
    # Check for Homebrew OpenMP installation
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel support")
    target_link_libraries(ncorr OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found - parallel features will be disabled")
endif()

# Set include directories (only for Mac Homebrew)
if(APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /opt/homebrew/include")
endif()

# Find OpenCV (with FetchContent fallback)
find_or_fetch_opencv()
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(ncorr ${OpenCV_LIBS})

# Find FFTW (with FetchContent fallback)
find_or_fetch_fftw()
if(FFTW_INCLUDE_DIR)
    target_include_directories(ncorr PRIVATE ${FFTW_INCLUDE_DIR})
    message(STATUS "FFTW include: ${FFTW_INCLUDE_DIR}")
endif()

# Compiler flags
target_compile_options(ncorr PRIVATE -Wall -Wextra)

# add -fpermissive flag to allow some non-standard code.
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# Disable debugging
ADD_DEFINITIONS(-DNDEBUG)

# Install library
INSTALL(TARGETS ncorr DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
INSTALL(FILES ${ncorr_h} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include)
