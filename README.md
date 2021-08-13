# Yinyang K-Means in Cuda
Yinyang kmeans algorithm implemented in CUDA

## Contents
- Build From Source
- Running the Compiled Program
- params.h
- Implemented Algorithms

## Build From Source
1. At the command prompt and in the desired directory, type `git clone https://github.com/ctaylor389/k_means_yinyang_gpu.git`
2. Edit the makefile with the corresponding cuda version (i.e. replace `-arch=compute_60 -code=sm_60` with `-arch=compute_<compute capability> -code=sm_<architecture>`) **Note**: program requires compute capability >= 6.0
3. Type `make` at the command line

## Running the Compiled Program
The standard usage pattern of the program is `./kmeans <algorithm> <filepath_to_dataset> <number_of_datapoints> <data_dimensionality> <number_of_clusters>` These positional arguments are required for a standard run of the program.
Optional arguments include:
-  **-t  <number_of_groups>** Must be <= number of clusters. Default = 20 for GPU, k/10 for CPU
-  **-c** count distance calculations
-  **-m <number_of_CPU_threads>** Default is 1
-  **-i <max_iterations_to_run>** Default is 1000
-  **-g <number_of_GPUs>** Default is 1
-  **-wc <write_to_filepath>** write final clusters to filepath
-  **-wa <write_to_filepath>** write final point assignments to filepath
-  **-wt <write_to_filepath>** write timing data to filepath
-  **-v** run validation tests
-  **-h** print help info

## Dataset Format
Dataset files must be of the following form for the program to read
```
<feature1>,<feature2>,<feature3>
<feature1>,<feature2>,<feature3>
```

## Implemented Algorithms
- Full Yinyang GPU - Full Yinyang algorithm implemented on the GPU utilizing all three filters. (FULLGPU)
- Full Yinyang CPU - Full Yinyang algorithm implemented on the CPU utilizing all three filters. (FULLCPU)
- Simplified Yinyang GPU - Simplified Yinyang algorithm implemented on the GPU utilizing two of the three filters. (SIMPLEGPU)
- Simplified Yinyang CPU - Simplified Yinyang algorithm implemented on the CPU utilizing two of the three filters. (SIMPLECPU)
- Super Simplified Yinyang GPU - Super simplified Yinyang algorithm implemented on the GPU utilizing one of three filters. (SUPERGPU)
- Super Simplified Yinyang CPU - Super simplified Yinyang algorithm implemented on the CPU utilizing one of three filters. (SUPERCPU)
- Lloyd GPU - Naive k-means implemented on the GPU (LLOYDGPU)
- Lloyd CPU - Naive k-means implemented on the GPU (LLOYDCPU)
