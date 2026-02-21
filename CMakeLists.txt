cmake_minimum_required(VERSION 3.13)
project("KMeans" LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_subdirectory(k_means_yinyang_cuda)