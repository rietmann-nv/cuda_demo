cmake_minimum_required(VERSION 3.9)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(population_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(cuda_demo
  main.cu
  )

set_property(TARGET cuda_demo PROPERTY CUDA_STANDARD 14)
