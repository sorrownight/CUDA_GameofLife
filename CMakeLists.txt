cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS -O3)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math -O3")
project(CudaTest LANGUAGES CXX CUDA)

add_executable(final gol.cu)

set_target_properties(final PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

include_directories("C:/Program Files (x86)/NVIDIA Corporation/Cg/include")
#target_link_libraries(final "C:/rogram Files (x86)/NVIDIA Corporation/Cg/lib.x64/glut32.lib")
