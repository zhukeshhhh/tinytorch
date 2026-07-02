cmake_minimum_required(VERSION 3.20)
project(tinytorch)
set(CMAKE_CXX_STANDARD 20)
option(USE_CUDA "Build with NVIDIA GPU Backend Support" OFF)

set(SOURCES
    src/core/matrix_factory.cpp
    src/core/tensor.cpp
    src/cpu/matrix_cpu.cpp
)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 120)
    list(APPEND SOURCES src/cuda/matrix_cuda.cu)
endif()

add_executable(tinytorch ${SOURCES} main.cpp)
add_executable(test      ${SOURCES} test.cpp)

foreach(TARGET tinytorch test)
    target_include_directories(${TARGET} PRIVATE include/)
    if(USE_CUDA)
        target_compile_definitions(${TARGET} PRIVATE USE_CUDA)
        target_link_libraries(${TARGET} PRIVATE curand)
    endif()
endforeach()