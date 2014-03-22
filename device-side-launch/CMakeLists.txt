# Copyright (c) 2014 NVIDIA Corporation
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

set(CMAKE_INSTALL_RPATH ${LIBNVVM_HOME})

if(CMAKE_SIZEOF_VOID_P STREQUAL "8")
  add_definitions(-DBUILD_64_BIT)
endif()

if (CMAKE_SIZEOF_VOID_P STREQUAL "8")
  if (WIN32)
    set (LIB_ARCH_SUFFIX "/x64")
  elseif (APPLE)
    set (LIB_ARCH_SUFFIX "")
  else ()
    set (LIB_ARCH_SUFFIX "64")
  endif()
else()
  if (WIN32)
    set (LIB_ARCH_SUFFIX "/Win32")
  else()
    set (LIB_ARCH_SUFFIX "")
  endif()
endif()

set_property(
   SOURCE dsl.c
   PROPERTY COMPILE_DEFINITIONS LIBCUDADEVRT="${CUDA_HOME}/lib${LIB_ARCH_SUFFIX}/libcudadevrt.a"
   )
   
set(CMAKE_INCLUDE_CURRENT_DIR YES)

add_executable(dsl dsl.c)

target_link_libraries(dsl ${NVVM_LIB} ${CUDA_LIB})

if (WIN32)
  set_target_properties(dsl PROPERTIES
    COMPILE_FLAGS "/wd4996")
else (WIN32)
  set_target_properties(dsl PROPERTIES
    LINK_FLAGS    "-Wl,-rpath,${LIBNVVM_RPATH}")
endif (WIN32)

install(TARGETS dsl DESTINATION bin)
install(FILES gpu32.ll
              gpu64.ll
        DESTINATION bin)
