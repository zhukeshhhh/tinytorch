cmake_minimum_required(VERSION 3.20)
project(tinytorch)

set(CMAKE_CXX_STANDARD 20)
option(USE_CUDA "Build with NVIDIA GPU Backend Support" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(
        SOURCES
        src/core/matrix_factory.cpp
        src/core/tensor.cpp
        src/cuda/matrix_cuda.cu
        src/cpu/matrix_cpu.cpp
        main.cpp
        )
    add_executable(tinytorch ${SOURCES})
    target_compile_definitions(tinytorch PRIVATE USE_CUDA)
    target_include_directories(tinytorch PRIVATE include/)

else()
    set(
        SOURCES
        src/core/matrix_factory.cpp
        src/core/tensor.cpp
        src/cpu/matrix_cpu.cpp
        main.cpp
        )
    add_executable(tinytorch ${SOURCES})
    target_include_directories(tinytorch PRIVATE include/)
endif()

