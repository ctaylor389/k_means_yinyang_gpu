#include <math.h>
#include <stdio.h>
#include "params.h"


// Atomic function overloads
__device__ void atomicMax(double const *address,
                          const double value);

__device__ void atomicMax(float const  *address, 
                          const float value);

// point assignment device kernels and functions
__global__ void assignPointsFull(PointInfo *pointInfo,
                                 CentInfo *centInfo,
                                 DTYPE *pointData,
                                 DTYPE *pointLwrs,
                                 DTYPE *centData,
                                 DTYPE *maxDriftArr,
                                 const int numPnt,
                                 const int numCent,
                                 const int numGrp,
                                 const int numDim);


__global__ void assignPointsSimple(PointInfo *pointInfo,
                                   CentInfo *centInfo,
                                   DTYPE *pointData,
                                   DTYPE *pointLwrs,
                                   DTYPE *centData,
                                   DTYPE *maxDriftArr,
                                   const int numPnt,
                                   const int numCent,
                                   const int numGrp,
                                   const int numDim);



__global__ void assignPointsSuper(PointInfo *pointInfo,
                                  CentInfo *centInfo,
                                  DTYPE *pointData,
                                  DTYPE *pointLwrs,
                                  DTYPE *centData,
                                  DTYPE *maxDrift,
                                  const int numPnt,
                                  const int numCent,
                                  const int numGrp,
                                  const int numDim);

__global__ void assignPointsLloyd(PointInfo *pointInfo,
                                  CentInfo *centInfo,
                                  DTYPE *pointData,
                                  DTYPE *centData,
                                  const int numPnt,
                                  const int numCent,
                                  const int numDim);

__device__ void pointCalcsFull(PointInfo *pointInfoPtr,
                               CentInfo *centInfo,
                               DTYPE *pointDataPtr,
                               DTYPE *pointLwrPtr,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               unsigned int *groupArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim);

__device__ void pointCalcsSimple(PointInfo *pointInfoPtr,
                               CentInfo *centInfo,
                               DTYPE *pointDataPtr,
                               DTYPE *pointLwrPtr,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               unsigned int *groupArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim);

// centroid calculation device kernels and functions
__global__ void clearCentCalcData(DTYPE *newCentSum,
                                  DTYPE *oldCentSum,
                                  unsigned int *newCentCount,
                                  unsigned int *oldCentCount,
                                  const int numCent,
                                  const int numDim);

__global__ void clearCentCalcDataLloyd(DTYPE *newCentSum,
                                       unsigned int *newCentCount,
                                       const int numCent,
                                       const int numDim);

__global__ void calcCentData(PointInfo *pointInfo,
                             CentInfo *centInfo,
                             DTYPE *pointData,
                             DTYPE *oldSums,
                             DTYPE *newSums,
                             unsigned int *oldCounts,
                             unsigned int *newCounts,
                             const int numPnt,
                             const int numDim);

__global__ void calcNewCentroids(PointInfo *pointInfo,
                                 CentInfo *centInfo,
                                 DTYPE *centData,
                                 DTYPE *oldSums,
                                 DTYPE *newSums,
                                 DTYPE *devMaxDriftArr,
                                 unsigned int *oldCounts,
                                 unsigned int *newCounts,
                                 const int numPnt,
                                 const int numDim);

__global__ void calcCentDataLloyd(PointInfo *pointInfo,
                                  DTYPE *pointData,
                                  DTYPE *newSums,
                                  unsigned int *newCounts,
                                  const int numPnt,
                                  const int numDim);

__global__ void calcNewCentroidsLloyd(PointInfo *pointInfo,
                                      CentInfo *centInfo,
                                      DTYPE *centData,
                                      DTYPE *newSums,
                                      unsigned int *newCounts,
                                      const int numCent,
                                      const int numDim);

__global__ void calcNewCentroidsAve(PointInfo *pointInfo,
                                      CentInfo *centInfo,
                                      DTYPE *centData,
                                      DTYPE *newSums,
                                      DTYPE *maxDriftArr,
                                      unsigned int *newCounts,
                                      const int numCent,
                                      const int numDim);

// helper kernels and functions
__global__ void warmup(unsigned int *tmp);

__global__ void checkConverge(PointInfo *pointInfo,
                              unsigned int *conFlag,
                              const int numPnt);

__global__ void initRunKernel(PointInfo *pointInfo,
                              CentInfo *centInfo,
                              DTYPE *pointData,
                              DTYPE *pointLwrs,
                              DTYPE *centData,
                              const int numPnt,
                              const int numCent,
                              const int numGrp,
                              const int numDim);

__global__ void clearDriftArr(DTYPE *devMaxDriftArr, const int numGrp);

__device__ DTYPE calcDis(DTYPE *vec1, DTYPE *vec2, const int numDim);

// overloads for counting # of distance calculations
__global__ void initRunKernel(PointInfo *pointInfo,
                              CentInfo *centInfo,
                              DTYPE *pointData,
                              DTYPE *pointLwrs,
                              DTYPE *centData,
                              const int numPnt,
                              const int numCent,
                              const int numGrp,
                              const int numDim,
                              unsigned long long int *calcCount);

__global__ void assignPointsFull(PointInfo *pointInfo,
                                 CentInfo *centInfo,
                                 DTYPE *pointData,
                                 DTYPE *pointLwrs,
                                 DTYPE *centData,
                                 DTYPE *maxDriftArr,
                                 const int numPnt,
                                 const int numCent,
                                 const int numGrp,
                                 const int numDim,
                                 unsigned long long int *calcCount);

__global__ void assignPointsSimple(PointInfo *pointInfo,
                                   CentInfo *centInfo,
                                   DTYPE *pointData,
                                   DTYPE *pointLwrs,
                                   DTYPE *centData,
                                   DTYPE *maxDriftArr,
                                   const int numPnt,
                                   const int numCent,
                                   const int numGrp,
                                   const int numDim,
                                   unsigned long long int *calcCount);

__global__ void assignPointsSuper(PointInfo *pointInfo,
                                  CentInfo *centInfo,
                                  DTYPE *pointData,
                                  DTYPE *pointLwrs,
                                  DTYPE *centData,
                                  DTYPE *maxDrift,
                                  const int numPnt,
                                  const int numCent,
                                  const int numGrp,
                                  const int numDim,
                                  unsigned long long int *calcCount);

__device__ void pointCalcsFull(PointInfo *pointInfoPtr,
                               CentInfo *centInfo,
                               DTYPE *pointDataPtr,
                               DTYPE *pointLwrPtr,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               unsigned int *groupArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim,
                               unsigned long long int *calcCount);

__device__ void pointCalcsSimple(PointInfo *pointInfoPtr,
                                 CentInfo *centInfo,
                                 DTYPE *pointDataPtr,
                                 DTYPE *pointLwrPtr,
                                 DTYPE *centData,
                                 DTYPE *maxDriftArr,
                                 unsigned int *groupArr,
                                 const int numPnt,
                                 const int numCent,
                                 const int numGrp,
                                 const int numDim,
                                 unsigned long long int *calcCount);

