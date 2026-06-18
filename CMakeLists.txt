cmake_minimum_required(VERSION 3.20)
project(tinytorch)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES
    core/matrix_factory.cpp
    core/tensor.cpp
    cpu/matrix_cpu.cpp
    main.cpp
)

add_executable(tinytorch ${SOURCES})

target_include_directories(tinytorch PRIVATE
    core/
    cpu/
)