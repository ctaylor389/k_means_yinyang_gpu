#ifndef KMEANSCPU_H
#define KMEANSCPU_H

#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "params.h"
#include "kmeansUtil.h"

double startFullOnCPU(PointInfo *pointInfo, 
                      CentInfo *centInfo, 
                      DTYPE *pointData, 
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numGrp, 
                      const int numDim,
                      const int numThread, 
                      const int maxIter, 
                      unsigned int *ranIter);

double startSimpleOnCPU(PointInfo *pointInfo, 
                        CentInfo *centInfo, 
                        DTYPE *pointData, 
                        DTYPE *centData,
                        const int numPnt,
                        const int numCent,
                        const int numGrp, 
                        const int numDim,
                        const int numThread,
                        const int maxIter, 
                        unsigned int *ranIter);

double startSuperOnCPU(PointInfo *pointInfo, 
                       CentInfo *centInfo, 
                       DTYPE *pointData,
                       DTYPE *centData, 
                       const int numPnt, 
                       const int numCent, 
                       const int numDim,
                       const int numThread, 
                       const int maxIter, 
                       unsigned int *ranIter);

double startLloydOnCPU(PointInfo *pointInfo, 
                       CentInfo *centInfo, 
                       DTYPE *pointData, 
                       DTYPE *centData, 
                       const int numPnt, 
                       const int numCent, 
                       const int numDim,
                       const int numThread,
                       const int maxIter, 
                       unsigned int *ranIter);

unsigned int checkConverge(PointInfo *pointInfo, 
                           const int numPnt);

void pointCalcsFullCPU(PointInfo *pointInfoPtr,
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

void pointCalcsSimpleCPU(PointInfo *pointInfoPtr,
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

void initPoints(PointInfo *pointInfo,
                CentInfo *centInfo, 
                DTYPE *pointData, 
                DTYPE *pointLwrs, 
                DTYPE *centData, 
                const int numPnt, 
                const int numCent, 
                const int numGrp, 
                const int numDim, 
                const int numThread);

void updateCentroids(PointInfo *pointInfo, 
                     CentInfo *centInfo, 
                     DTYPE *pointData,
                     DTYPE *centData, 
                     DTYPE *maxDriftArr,
                     const int numPnt, 
                     const int numCent, 
                     const int numGrp, 
                     const int numDim, 
                     const int numThread);



#endif
