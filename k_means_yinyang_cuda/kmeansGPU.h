#ifndef GPU_H
#define GPU_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <unistd.h>
#include "params.h"
#include "kmeansKernel.h"
#include "kmeansUtil.h"
#include "omp.h"


double startFullOnGPU(PointInfo *pointInfo,
                      CentInfo *centInfo,
                      DTYPE *pointData,
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numGrp,
                      const int numDim,
                      const int maxIter,
                      unsigned int *ranIter);

double startSimpleOnGPU(PointInfo *pointInfo,
                        CentInfo *centInfo,
                        DTYPE *pointData,
                        DTYPE *centData,
                        const int numPnt,
                        const int numCent,
                        const int numGrp,
                        const int numDim,
                        const int maxIter,
                        unsigned int *ranIter);

double startSimpleOnGPU(PointInfo *pointInfo,
                        CentInfo *centInfo,
                        DTYPE *pointData,
                        DTYPE *centData,
                        const int numPnt,
                        const int numCent,
                        const int numGrp,
                        const int numDim,
                        const int maxIter,
                        const int numGPU,
                        unsigned int *ranIter);

void calcWeightedMeans(CentInfo *newCentInfo,
                       CentInfo **allCentInfo,
                       DTYPE *newCentData,
                       DTYPE *oldCentData,
                       DTYPE **allCentData,
                       DTYPE *newMaxDriftArr,
                       const int numCent,
                       const int numGrp,
                       const int numDim,
                       const int numGPU);

double startSuperOnGPU(PointInfo *pointInfo,
                        CentInfo *centInfo,
                        DTYPE *pointData,
                        DTYPE *centData,
                        const int numPnt,
                        const int numCent,
                        const int numDim,
                        const int maxIter,
                        unsigned int *ranIter);

double startLloydOnGPU(PointInfo *pointInfo,
                       CentInfo *centInfo,
                       DTYPE *pointData,
                       DTYPE *centData,
                       const int numPnt,
                       const int numCent,
                       const int numDim,
                       const int maxIter,
                       unsigned int *ranIter);

DTYPE *storeDataOnGPU(DTYPE *data,
                      const int numVec,
                      const int numFeat);

PointInfo *storePointInfoOnGPU(PointInfo *pointInfo,
                               const int numPnt);

CentInfo *storeCentInfoOnGPU(CentInfo *centInfo,
                             const int numCent);

void warmupGPU();

void warmupGPU(const int numGPU);


double startFullOnGPU(PointInfo *pointInfo,
                    CentInfo *centInfo,
                    DTYPE *pointData,
                    DTYPE *centData,
                    const int numPnt,
                    const int numCent,
                    const int numGrp,
                    const int numDim,
                    const int maxIter,
                    unsigned int *ranIter,
                    unsigned long long int *countPtr);

double startSimpleOnGPU(PointInfo *pointInfo,
                      CentInfo *centInfo,
                      DTYPE *pointData,
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numGrp,
                      const int numDim,
                      const int maxIter,
                      const int numGPU,
                      unsigned int *ranIter,
                      unsigned long long int *countPtr);

double startSuperOnGPU(PointInfo *pointInfo,
                     CentInfo *centInfo,
                     DTYPE *pointData,
                     DTYPE *centData,
                     const int numPnt,
                     const int numCent,
                     const int numDim,
                     const int maxIter,
                     unsigned int *ranIter,
                     unsigned long long int *countPtr);

double startLloydOnGPU(PointInfo *pointInfo,
                     CentInfo *centInfo,
                     DTYPE *pointData,
                     DTYPE *centData,
                     const int numPnt,
                     const int numCent,
                     const int numDim,
                     const int maxIter,
                     unsigned int *ranIter,
                     unsigned long long int *countPtr);

#endif
