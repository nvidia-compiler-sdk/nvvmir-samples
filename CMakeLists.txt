# Copyright (c) 2014-2022 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.10.0)

project(libnvvm-samples)

# Initialize the CUDA Toolkit search using CUDA_HOME if the user specified it.
if (DEFINED ENV{CUDA_HOME})
  if (${CMAKE_VERSION} VERSION_LESS 3.18.0) 
    set(CUDA_TOOLKIT_ROOT_DIR  "$ENV{CUDA_HOME}" CACHE PATH "Path to CUDA Toolkit.")
  else ()  # cmake 3.17 introduced FindCUDAToolkit
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "Path to CUDA Toolkit.")
  endif ()
endif ()

# Locate the root directory of the CUDA Toolkit.
# cmake 3.18 introduces the CUDAToolkit_LIBRARY_ROOT variable.
# This will update CUDA_HOME to a value discovered by find_package.
if (${CMAKE_VERSION} VERSION_LESS 3.18.0) 
  find_package(CUDA REQUIRED)
  set(CUDA_HOME "${CUDA_TOOLKIT_ROOT_DIR}")
  get_filename_component(CUDA_LIB_ROOT "${CUDA_cudart_static_LIBRARY}" DIRECTORY)
  find_library(CUDA_LIB NAMES cuda cuda.lib PATHS "${CUDA_LIB_ROOT}" "${CUDA_LIB_ROOT}/stubs")
  find_library(CUDADEVRT_LIB cudadevrt PATHS "${CUDA_LIB_ROOT}" "${CUDA_LIB_ROOT}/stubs")
  include_directories("${CUDA_INCLUDE_DIRS}")
else () # Else, we're using cmake versions >= 3.18.
  cmake_policy(SET CMP0074 NEW) # Use CUDAToolkit_ROOT as a cmake prefix.
  find_package(CUDAToolkit REQUIRED)
  set(CUDA_LIB "${CUDA_cuda_driver_LIBRARY}")
  include_directories("${CUDAToolkit_INCLUDE_DIRS}")
  get_filename_component(CUDA_HOME "${CUDAToolkit_BIN_DIR}" DIRECTORY)
  find_library(CUDADEVRT_LIB cudadevrt PATH "${CUDAToolkit_LIBRARY_DIR}")
endif ()
message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")
message(STATUS "Using CUDA_LIB:  ${CUDA_LIB}")
message(STATUS "Using CUDADEVRT_LIB: ${CUDADEVRT_LIB}")
if (("${CUDA_HOME}" STREQUAL "") OR ("${CUDA_LIB}" STREQUAL ""))
  message(FATAL_ERROR "Failed to locate paths to the CUDA toolkit and nvcc.")
endif ()

# Find the nvvm directory in the toolkit.
find_file(LIBNVVM_HOME nvvm PATHS "$ENV{LIBNVVM_HOME}" "${CUDA_HOME}")
message(STATUS "Using LIBNVVM_HOME: ${LIBNVVM_HOME}")

# Find libNVVM and nvvm.h.
# (Linux: nvvm/lib64, windows: nvvm/lib/x64)
find_library(NVVM_LIB nvvm PATHS "${LIBNVVM_HOME}/lib64" "${LIBNVVM_HOME}/lib/x64")
find_file(NVVM_H nvvm.h PATH "${LIBNVVM_HOME}/include")
get_filename_component(NVVM_INCLUDE_DIR ${NVVM_H} DIRECTORY)
include_directories(${NVVM_INCLUDE_DIR})
message(STATUS "Using libnvvm header:      ${NVVM_H}")
message(STATUS "Using libnvvm header path: ${NVVM_INCLUDE_DIR}")
message(STATUS "Using libnvvm library:     ${NVVM_LIB}")

# Find libcudadevrt.
find_library(CUDADEVRT_LIB cudadevrt PATHS "${CUDA_LIB_ROOT}")
message(STATUS "Using CUDADEVRT_LIB: ${CUDADEVRT_LIB}")
if (NOT CUDADEVRT_LIB)
  message(FATAL_ERROR "Failed to locate the path to libcudadevrt.")
endif ()

# Set the rpath to libnvvm.
find_path(LIBNVVM_RPATH lib lib64 PATHS "$ENV{LIBNVVM_HOME}" "${CUDA_HOME}")
get_filename_component(LIBNVVM_RPATH ${NVVM_LIB} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${LIBNVVM_RPATH}") 
message(STATUS "Using rpath: ${CMAKE_INSTALL_RPATH}")

# On Windows, locate the nvvm.dll so we can install it.
if (WIN32)
  find_file(NVVM_DLL nvvm64_40_0.dll PATHS "${LIBNVVM_HOME}/bin")
  if(NOT NVVM_DLL)
    message(FATAL_ERROR "Found nvvm .h/.lib, but not .dll")
  endif()
  install(FILES ${NVVM_DLL} DESTINATION bin)
endif (WIN32)

add_definitions(-DLIBDEVICE_MAJOR_VERSION=1)
add_definitions(-DLIBDEVICE_MINOR_VERSION=0)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common/include")

# Samples
add_subdirectory(device-side-launch)
add_subdirectory(uvmlite)
