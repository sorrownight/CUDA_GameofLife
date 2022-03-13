cmake_minimum_required(VERSION 3.20)


set(CMAKE_CUDA_STANDARD 14)

project(CudaTest LANGUAGES CXX CUDA)

add_executable(final firstItr.cu)

set_target_properties(final PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
