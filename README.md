# k_means_yinyang_gpu
Yinyang kmeans algorithm implemented in CUDA

## Contents
- Build From Source
- Running the Compiled Program
- params.h
- Implemented Algorithms

## Build From Source
1. At the command prompt and in the desired directory, type `git clone https://github.com/ctaylor389/k_means_yinyang_gpu.git`
2. Edit params.h header file to align with desired dataset (see params.h)
3. Edit the makefile with the corresponding cuda version (i.e. replace `-arch=compute_60 -code=sm_60` with `-arch=compute_<compute capability> -code=sm_<architecture>`) **Note**: program requires compute capability >= 6.0
4. Type `make` at the command line

## Running the Compiled Program
The usage pattern of the program is `./kmeans <path/to/dataset> <#_of_feature_vectors> <data_dimensionality> <algorithm_code> <verbose_flag> <path/to/writeout/timedata>`
- **path to dataset** - the absolute or relative path to the dataset in plain text in the following format wherein each feature vector is seperated by a new line and each feature is seperated by a comma.
```
<feature1>,<feature2>,<feature3>
<feature1>,<feature2>,<feature3>
```
- **\# of feature vectors** - total number of feature vectors in the input dataset
- **data dimensionality** - dimensionality of the input dataset
- **algorithm code** - what version of kmeans to run. See Implemented Algorithms
- **verbose flag** - 0 or 1 for counting and outputting total distance calculations performed.
- **path to write out time data** - the absolute or relative path to write time data out to. Providing NULL will not write out time data. 


## params.h
The params.h file must be properly configured *before* building with the makefile provided in the repository.
Main features of the params.h file include:
- **GROUPTYPE** - valid options include NGROUP or NGROUPCPU. One should define NGROUP if they want to use any of the GPU implementations and NGROUPCPU if they want to use any of the CPU implementations.
- **DTYPE** - valid options include double and float. This defines which level of precision the program will use when performaing kmeans.
- **BLOCKSIZE** - the number of threads in a single block.
- **NDIM** - the dimensionality of the dataset.
- **NPOINT** - the total number of feature vectors in the dataset.
- **NCLUST** - the number of clusters to make
- **NGROUP** - the number of groups cluster centers are partitioned into on GPU implementations. Default value for the GPU is 20 or the number of clusters if the total cluster count is less than 20.
- **NGROUPCPU** - the number of groups cluster centers are partitioned into on CPU implementations. Default value for the CPU is one tenth of the total cluster count (i.e. *k*/10 where *k* is the total cluster count)
- **MAXITER** - the maximum number of iterations kmeans can run.
- **SEED** - RNG seed for random generations of cluster centers.
- **NTHREADS** - number of threads to use in parallel CPU implmentations.
- **NRUNS** - number of times to run kmeans (for use when running time trials)

## Implemented Algorithms
- Full Yinyang GPU - Full Yinyang algorithm implemented on the GPU utilizing all three filters. (Algorithm code 0)
- Full Yinyang CPU - Full Yinyang algorithm implemented on the CPU utilizing all three filters. (Algorithm code 1)
- Simplified Yinyang GPU - Simplified Yinyang algorithm implemented on the GPU utilizing two of the three filters. (Algorithm code 2)
- Simplified Yinyang CPU - Simplified Yinyang algorithm implemented on the CPU utilizing two of the three filters. (Algorithm code 3)
- Super Simplified Yinyang GPU - Super simplified Yinyang algorithm implemented on the GPU utilizing one of three filters. (Algorithm code 4)
- Super Simplified Yinyang CPU - Super simplified Yinyang algorithm implemented on the CPU utilizing one of three filters. (Algorithm code 5)
- Lloyd GPU - Naive k-means implemented on the GPU (Algorithm code 6)
- Lloyd CPU - Naive k-means implemented on the GPU (Algorithm code 7)
