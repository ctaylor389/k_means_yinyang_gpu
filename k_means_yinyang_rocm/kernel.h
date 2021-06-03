#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include "params.h"


// Atomic function overloads
__device__ void atomicMax(double const *address,
                          const double value);

__device__ void atomicMax(float const  *address, 
                          const float value);

// point assignment device kernels and functions
__global__ void assignPointsFull(point *dataset,
                                 cent *centroidDataset,
                                 DTYPE *maxDriftArr);


__global__ void assignPointsSimple(point *dataset,
                                   cent *centroidDataset,
                                   DTYPE *maxDriftArr);



__global__ void assignPointsSuper(point *dataset,
                                  cent *centroidDataset,
                                  DTYPE *maxDrift);

__global__ void assignPointsLloyd(point *dataset,
                                  cent *centroidDataset);

__device__ void pointCalcsFull(point *pointPtr,
                               cent *centroidDataset,
                               unsigned int *groupArr,
                               DTYPE *maxDriftArr);

__device__ void pointCalcsFullAlt(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr);

__device__ void pointCalcsSimple(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr);

// centroid calculation device kernels and functions
__global__ void clearCentCalcData(vector *newCentSum,
                                  vector *oldCentSum,
                                  unsigned int *newCentCount,
                                  unsigned int *oldCentCount);

__global__ void clearCentCalcDataLloyd(vector *newCentSum,
                                       unsigned int *newCentCount);

__global__ void updateCentroidsKernel(point *dataset,
                                      cent *centroidDataset,
                                      DTYPE *maxDriftArr);

__global__ void updateCentroidsLloyd(point *dataset,
                                     cent *centroidDataset,
                                     DTYPE *maxDriftArr);

__global__ void calcCentData(point *dataset,
                             cent *centroidDataset,
                             vector *oldSums,
                             vector *newSums,
                             unsigned int *oldCounts,
                             unsigned int *newCounts);

__global__ void calcNewCentroids(point *dataset,
                                 cent *centroidDataset,
                                 DTYPE *maxDriftArr,
                                 vector *oldSums,
                                 vector *newSums,
                                 unsigned int *oldCounts,
                                 unsigned int *newCounts);

__global__ void calcCentDataLloyd(point *dataset,
                                  cent *centroidDataset,
                                  vector *newSums,
                                  unsigned int *newCounts);

__global__ void calcNewCentroidsLloyd(point *dataset,
                                      cent *centroidDataset,
                                      vector *newSums,
                                      unsigned int *newCounts);

__global__ void calcNewCentroidsAve(point *dataset,
                                    cent *centroidDataset,
                                    vector *newSums,
                                    unsigned int *newCounts,
                                    DTYPE *maxDriftArr);

// helper kernels and functions
__global__ void warmup(unsigned int *tmp);

__global__ void checkConverge(point * dataset,
                              unsigned int * conFlag);

__global__ void initRunKernel(point *dataset,
                              cent *centroidDataset);

__global__ void clearDriftArr(DTYPE *devMaxDriftArr);

__device__ DTYPE calcDis(vector *vec1,
                         vector *vec2);

// overloads for counting # of distance calculations
__global__ void initRunKernel(point *dataset,
                              cent *centroidDataset,
                              unsigned long long int *calcCount);

__global__ void assignPointsFull(point *dataset,
                                 cent *centroidDataset,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount);

__global__ void assignPointsSimple(point *dataset,
                                   cent *centroidDataset,
                                   DTYPE *maxDriftArr,
                                   unsigned long long int *calcCount);

__global__ void assignPointsSuper(point *dataset,
                                  cent *centroidDataset,
                                  DTYPE *maxDrift,
                                  unsigned long long int *calcCount);

__device__ void pointCalcsFull(point *pointPtr,
                               cent *centroidDataset,
                               unsigned int *groupArr,
                               DTYPE *maxDriftArr,
                               unsigned long long int *calcCount);

__device__ void pointCalcsFullAlt(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount);

__device__ void pointCalcsSimple(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount);

