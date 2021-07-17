#include "hip/hip_runtime.h"
#include "kmeansGPU.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
 if (code != hipSuccess)
 {
    fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort) exit(code);
 }
}

/////////////////////////////////////////////
// Host functions for calling CUDA kernels //
/////////////////////////////////////////////

double startFullOnGPU(PointInfo *pointInfo, 
                    CentInfo *centInfo, 
                    DTYPE *pointData,
                    DTYPE *centData, 
                    const int numPnt, 
                    const int numCent,
                    const int numGrp, 
                    const int numDim, 
                    const int maxIter, 
                    unsigned int *ranIter)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;
  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;
  int oldPosSize = sizeof(DTYPE)*numDim*BLOCKSIZE;
  
  int index = 1;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);

  // group centroids
  groupCent(centInfo, centData, numCent, numGrp, numDim);
  
  // create lower bound data on host
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numGrp);
  for(int i = 0; i < numPnt * numGrp; i++)
  {
    pointLwrs[i] = INFINITY;
  }

  // store dataset on device
  PointInfo *devPointInfo;
  DTYPE *devPointData;
  DTYPE *devPointLwrs;

  devPointInfo = storePointInfoOnGPU(pointInfo, numPnt);
  devPointData = storeDataOnGPU(pointData, numPnt, numDim);
  devPointLwrs = storeDataOnGPU(pointLwrs, numPnt, numGrp);

  // store centroids on device
  CentInfo *devCentInfo;
  DTYPE *devCentData;
  
  devCentInfo = storeCentInfoOnGPU(centInfo, numCent);
  devCentData = storeDataOnGPU(centData, numCent, numDim);


  DTYPE *devMaxDriftArr = NULL;
  hipMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  hipMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  hipMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  hipMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  hipMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  hipMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(hipMemcpy(devConFlag, hostConFlagPtr, 
                        sizeof(unsigned int),hipMemcpyHostToDevice));


  hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr, numGrp);

  // do single run of naive kmeans for initial centroid assignments
  hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo, 
                                       devCentInfo,
                                       devPointData,
                                       devPointLwrs,
                                       devCentData,
                                       numPnt,
                                       numCent,
                                       numGrp,
                                       numDim);
  

  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;
    
    gpuErrchk(hipMemcpy(devConFlag,hostConFlagPtr, 
                           sizeof(unsigned int),hipMemcpyHostToDevice));

    // clear maintained data on device
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr, numGrp);

    // calculate data necessary to make new centroids
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDriftArr,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);

    hipLaunchKernelGGL(assignPointsFull, dim3(NBLOCKS), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo,devCentInfo,
                                                         devPointData,devPointLwrs,
                                                         devCentData,devMaxDriftArr,
                                                         numPnt,numCent,numGrp,numDim);

    hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(hipMemcpy(hostConFlagPtr,
        devConFlag, sizeof(unsigned int),
                    hipMemcpyDeviceToHost));
  }
  // calc final centroids (for matching results with lloyds)
  hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDriftArr,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
  hipDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(hipMemcpy(pointInfo, devPointInfo,sizeof(PointInfo)*numPnt,hipMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData,devCentData,sizeof(DTYPE)*numDim*numCent,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  hipFree(devPointInfo);
  hipFree(devPointData);
  hipFree(devPointLwrs);
  hipFree(devCentInfo);
  hipFree(devCentData);
  hipFree(devMaxDriftArr);
  hipFree(devNewCentSum);
  hipFree(devOldCentSum);
  hipFree(devNewCentCount);
  hipFree(devOldCentCount);
  hipFree(devConFlag);
  
  free(pointLwrs);
  
  endTime = omp_get_wtime();
  return endTime - startTime;
}

double startSimpleOnGPU(PointInfo *pointInfo,
                      CentInfo *centInfo,
                      DTYPE *pointData,
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numGrp,
                      const int numDim,
                      const int maxIter,
                      unsigned int *ranIter)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;
  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;
  int oldPosSize = sizeof(DTYPE)*numDim*BLOCKSIZE;
  
  int index = 1;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);


  // group centroids
  groupCent(centInfo, centData, numCent, numGrp, numDim);
  
  // create lower bound data on host
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numGrp);
  for(int i = 0; i < numPnt * numGrp; i++)
  {
    pointLwrs[i] = INFINITY;
  }

  // store dataset on device
  PointInfo *devPointInfo;
  DTYPE *devPointData;
  DTYPE *devPointLwrs;

  devPointInfo = storePointInfoOnGPU(pointInfo, numPnt);
  devPointData = storeDataOnGPU(pointData, numPnt, numDim);
  devPointLwrs = storeDataOnGPU(pointLwrs, numPnt, numGrp);

  // store centroids on device
  CentInfo *devCentInfo;
  DTYPE *devCentData;
  
  devCentInfo = storeCentInfoOnGPU(centInfo, numCent);
  devCentData = storeDataOnGPU(centData, numCent, numDim);

  DTYPE *devMaxDriftArr = NULL;
  hipMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  hipMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  hipMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  hipMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  hipMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  hipMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(hipMemcpy(devConFlag,hostConFlagPtr,sizeof(unsigned int),hipMemcpyHostToDevice));


  hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr, numGrp);

  // do single run of naive kmeans for initial centroid assignments
  hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo, 
                                       devCentInfo,
                                       devPointData,
                                       devPointLwrs,
                                       devCentData,
                                       numPnt,
                                       numCent,
                                       numGrp,
                                       numDim);

  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;
    
    gpuErrchk(hipMemcpy(devConFlag,hostConFlagPtr, 
                           sizeof(unsigned int),hipMemcpyHostToDevice));

    // clear maintained data on device
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr, numGrp);


    // calculate data necessary to make new centroids
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDriftArr,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);
    
    hipLaunchKernelGGL(assignPointsSimple, dim3(NBLOCKS), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo,devCentInfo,
                                                           devPointData,devPointLwrs,
                                                           devCentData,devMaxDriftArr,
                                                           numPnt,numCent,numGrp,numDim);

    hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,
                                         devConFlag,
                                         numPnt);
    index++;
    gpuErrchk(hipMemcpy(hostConFlagPtr, devConFlag, 
                           sizeof(unsigned int), hipMemcpyDeviceToHost));
  }
  hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDriftArr,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
  hipDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(hipMemcpy(pointInfo, devPointInfo,
                       sizeof(PointInfo)*numPnt,hipMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData,devCentData,
                       sizeof(DTYPE)*numDim*numCent,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  hipFree(devPointInfo);
  hipFree(devPointData);
  hipFree(devPointLwrs);
  hipFree(devCentInfo);
  hipFree(devCentData);
  hipFree(devMaxDriftArr);
  hipFree(devNewCentSum);
  hipFree(devOldCentSum);
  hipFree(devNewCentCount);
  hipFree(devOldCentCount);
  hipFree(devConFlag);
  
  free(pointLwrs);
  
  endTime = omp_get_wtime();
  return endTime - startTime;
}

double startSuperOnGPU(PointInfo *pointInfo,
                     CentInfo *centInfo,
                     DTYPE *pointData, 
                     DTYPE *centData,
                     const int numPnt,
                     const int numCent,
                     const int numDim,
                     const int maxIter,
                     unsigned int *ranIter)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;
  int oldPosSize = sizeof(DTYPE)*numDim*BLOCKSIZE;
  
  int index = 1;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);


  // group centroids
  for(int j = 0; j < numCent; j++)
  {
    centInfo[j].groupNum = 0;
  }
  
  // create lower bound data on host
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt);
  for(int i = 0; i < numPnt; i++)
  {
    pointLwrs[i] = INFINITY;
  }

  // store dataset on device
  PointInfo *devPointInfo;
  DTYPE *devPointData;
  DTYPE *devPointLwrs;

  devPointInfo = storePointInfoOnGPU(pointInfo, numPnt);
  devPointData = storeDataOnGPU(pointData, numPnt, numDim);
  devPointLwrs = storeDataOnGPU(pointLwrs, numPnt, 1);

  // store centroids on device
  CentInfo *devCentInfo;
  DTYPE *devCentData;
  
  devCentInfo = storeCentInfoOnGPU(centInfo, numCent);
  devCentData = storeDataOnGPU(centData, numCent, numDim);

  DTYPE *devMaxDrift = NULL;
  hipMalloc(&devMaxDrift, sizeof(DTYPE));

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  hipMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  hipMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  hipMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  hipMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  hipMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(hipMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),hipMemcpyHostToDevice));


  hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDrift, 1);

  // do single run of naive kmeans for initial centroid assignments
  hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo, 
                                       devCentInfo,
                                       devPointData,
                                       devPointLwrs,
                                       devCentData,
                                       numPnt,
                                       numCent,
                                       1,
                                       numDim);
  

  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;
    
    gpuErrchk(hipMemcpy(devConFlag, hostConFlagPtr,
                           sizeof(unsigned int), hipMemcpyHostToDevice));

    // clear maintained data on device
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDrift, 1);

    // calculate data necessary to make new centroids
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDrift,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);
    
    hipLaunchKernelGGL(assignPointsSuper, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                              devPointData,devPointLwrs,
                                              devCentData,devMaxDrift,
                                              numPnt,numCent,1,numDim);

    hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,
                                         devConFlag,
                                         numPnt);
    index++;
    gpuErrchk(hipMemcpy(hostConFlagPtr, devConFlag,
                         sizeof(unsigned int), hipMemcpyDeviceToHost));
  }
  hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), oldPosSize, 0, devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDrift,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
  hipDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(hipMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt, hipMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData,
                         sizeof(DTYPE)*numDim*numCent, hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  hipFree(devPointInfo);
  hipFree(devPointData);
  hipFree(devPointLwrs);
  hipFree(devCentInfo);
  hipFree(devCentData);
  hipFree(devMaxDrift);
  hipFree(devNewCentSum);
  hipFree(devOldCentSum);
  hipFree(devNewCentCount);
  hipFree(devOldCentCount);
  hipFree(devConFlag);
  
  free(pointLwrs);
  
  endTime = omp_get_wtime();
  return endTime - startTime;
}

double startLloydOnGPU(PointInfo *pointInfo,
                     CentInfo *centInfo,
                     DTYPE *pointData, 
                     DTYPE *centData,
                     const int numPnt,
                     const int numCent,
                     const int numDim,
                     const int maxIter,
                     unsigned int *ranIter)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;

  int index = 0;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);

  // store dataset on device
  PointInfo *devPointInfo;
  DTYPE *devPointData;

  devPointInfo = storePointInfoOnGPU(pointInfo, numPnt);
  devPointData = storeDataOnGPU(pointData, numPnt, numDim);

  // store centroids on device
  CentInfo *devCentInfo;
  DTYPE *devCentData;
  
  devCentInfo = storeCentInfoOnGPU(centInfo, numCent);
  devCentData = storeDataOnGPU(centData, numCent, numDim);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  hipMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  hipMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  hipMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(hipMemcpy(devConFlag, hostConFlagPtr,
                         sizeof(unsigned int), hipMemcpyHostToDevice));


  hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                 devNewCentCount,
                                                 numCent,
                                                 numDim);


  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;
    
    gpuErrchk(hipMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(assignPointsLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,
                                              devCentInfo,
                                              devPointData,
                                              devCentData,
                                              numPnt,
                                              numCent,
                                              numDim);

    hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                   devNewCentCount,
                                                   numCent,
                                                   numDim);
    // calculate data necessary to make new centroids
    hipLaunchKernelGGL(calcCentDataLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,
                                              devPointData,
                                              devNewCentSum,
                                              devNewCentCount,
                                              numPnt,
                                              numDim);

    // make new centroids
    hipLaunchKernelGGL(calcNewCentroidsLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,
                                                  devCentInfo,
                                                  devCentData,
                                                  devNewCentSum,
                                                  devNewCentCount,
                                                  numCent,
                                                  numDim);

    hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(hipMemcpy(hostConFlagPtr,devConFlag,
                           sizeof(unsigned int),hipMemcpyDeviceToHost));
  }
  hipDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(hipMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt,hipMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData,devCentData,
                         sizeof(DTYPE)*numDim*numCent,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  hipFree(devPointInfo);
  hipFree(devPointData);
  hipFree(devCentInfo);
  hipFree(devCentData);
  hipFree(devNewCentSum);
  hipFree(devNewCentCount);
  hipFree(devConFlag);
  
  endTime = omp_get_wtime();
  return endTime - startTime;
}

PointInfo *storePointInfoOnGPU(PointInfo *pointInfo, 
                               const int numPnt)
{
  PointInfo *devPointInfo = NULL;
  gpuErrchk(hipMalloc(&devPointInfo, sizeof(PointInfo)*numPnt));
  gpuErrchk(hipMemcpy(devPointInfo, pointInfo, 
                         sizeof(PointInfo)*numPnt, hipMemcpyHostToDevice));
  return devPointInfo;
}

CentInfo *storeCentInfoOnGPU(CentInfo *centInfo, 
                             const int numCent)
{
  CentInfo *devCentInfo = NULL;
  gpuErrchk(hipMalloc(&devCentInfo, sizeof(CentInfo) * numCent));
  gpuErrchk(hipMemcpy(devCentInfo, centInfo, 
                         sizeof(CentInfo)*numCent, hipMemcpyHostToDevice));
  return devCentInfo;
}

DTYPE *storeDataOnGPU(DTYPE *data,
                      const int numVec,
                      const int numFeat)
{
  DTYPE *devData = NULL;
  gpuErrchk(hipMalloc(&devData, sizeof(DTYPE) * numVec * numFeat));
  gpuErrchk(hipMemcpy(devData, data, 
                         sizeof(DTYPE)*numVec*numFeat, hipMemcpyHostToDevice));
  return devData;
}

void warmupGPU()
{
  unsigned int *dev_tmp;
  unsigned int *tmp;
  tmp = (unsigned int*)malloc(sizeof(unsigned int));
  *tmp = 0;
  hipMalloc((unsigned int**)&dev_tmp, sizeof(unsigned int));

  hipLaunchKernelGGL(warmup, dim3(1), dim3(256), 0, 0, dev_tmp);

  hipMemcpy(tmp, dev_tmp, sizeof(unsigned int), hipMemcpyDeviceToHost);

  hipDeviceSynchronize();

  hipFree(dev_tmp);
}
