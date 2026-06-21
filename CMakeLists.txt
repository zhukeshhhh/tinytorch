cmake_minimum_required(VERSION 3.20)
project(tinytorch)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
    src/core/matrix_factory.cpp
    src/core/tensor.cpp
    src/cpu/matrix_cpu.cpp
    main.cpp
)

add_executable(tinytorch ${SOURCES})

target_include_directories(tinytorch PRIVATE include/)