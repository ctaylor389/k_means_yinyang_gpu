#include "kmeansGPU.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
 if (code != cudaSuccess)
 {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
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
  cudaMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentData = NULL;
  cudaMalloc(&devOldCentData, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  cudaMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                        sizeof(unsigned int),cudaMemcpyHostToDevice));


  clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);

  // do single run of naive kmeans for initial centroid assignments
  initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
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

    gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                           sizeof(unsigned int),cudaMemcpyHostToDevice));

    // clear maintained data on device
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);

    // calculate data necessary to make new centroids
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

    assignPointsFull<<<NBLOCKS, BLOCKSIZE, grpLclSize>>>(devPointInfo,devCentInfo,
                                                         devPointData,devPointLwrs,
                                                         devCentData,devMaxDriftArr,
                                                         numPnt,numCent,numGrp,numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr,
        devConFlag, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost));
  }
  // calc final centroids (for matching results with lloyds)
  calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,sizeof(PointInfo)*numPnt,cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData,devCentData,sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devPointLwrs);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devMaxDriftArr);
  cudaFree(devNewCentSum);
  cudaFree(devOldCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devOldCentCount);
  cudaFree(devConFlag);

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
  cudaMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);
  
  DTYPE *devOldCentData = NULL;
  cudaMalloc(&devOldCentData, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  cudaMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,sizeof(unsigned int),cudaMemcpyHostToDevice));


  clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);

  // do single run of naive kmeans for initial centroid assignments
  initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
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

    gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                           sizeof(unsigned int),cudaMemcpyHostToDevice));

    // clear maintained data on device
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);


    // calculate data necessary to make new centroids
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

    assignPointsSimple<<<NBLOCKS, BLOCKSIZE, grpLclSize>>>(devPointInfo,devCentInfo,
                                                           devPointData,devPointLwrs,
                                                           devCentData,devMaxDriftArr,
                                                           numPnt,numCent,numGrp,numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
                                         devConFlag,
                                         numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr, devConFlag,
                           sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,
                       sizeof(PointInfo)*numPnt,cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData,devCentData,
                       sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devPointLwrs);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devMaxDriftArr);
  cudaFree(devNewCentSum);
  cudaFree(devOldCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devOldCentCount);
  cudaFree(devConFlag);

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
                      const int numGPU,
                      unsigned int *ranIter)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  int numPnts[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    if (numPnt % numGPU != 0 && i == numGPU-1)
    {
      numPnts[i] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[i] = numPnt / numGPU;
    }

  }

  // variable initialization
  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    hostConFlagArr[i] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    hostConFlagPtrArr[i] = &hostConFlagArr[i];
  }

  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;

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
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];
  DTYPE *devPointLwrs[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    cudaSetDevice(i);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[i], sizeof(PointInfo)*(numPnts[i])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[i],
                         pointInfo+(i*numPnt/numGPU),
                         (numPnts[i])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[i], sizeof(DTYPE) * numPnts[i] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[i],
                         pointData+((i*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[i]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[i], sizeof(DTYPE) * numPnts[i] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[i],
                         pointLwrs+((i*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[i]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[i], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[i], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[i],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[i], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[i],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devMaxDriftArr[i], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentSum[i], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentSum[i], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentCount[i], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentCount[i], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devConFlagArr[i], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[i],
              hostConFlagPtrArr[i], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[i],
                                              devOldCentSum[i],
                                              devNewCentCount[i],
                                              devOldCentCount[i],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[i],
                                         devCentInfo[i],
                                         devPointData[i],
                                         devPointLwrs[i],
                                         devCentData[i],
                                         numPnts[i],
                                         numCent,
                                         numGrp,
                                         numDim);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (int i = 0; i < numGPU; i++)
  {
    allCentInfo[i] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (int i = 0; i < numGPU; i++)
  {
    allCentData[i] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
  }

  CentInfo *newCentInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);

  DTYPE *newCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);
  for (int i = 0; i < numCent; i++)
  {
    for (int j = 0; j < numDim; j++)
    {
      newCentData[(i * numDim) + j] = 0;
    }
  }

  DTYPE *oldCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  DTYPE *newMaxDriftArr;
  newMaxDriftArr=(DTYPE *)malloc(sizeof(DTYPE)*numGrp);
  for (int i = 0; i < numGrp; i++)
  {
    newMaxDriftArr[i] = 0.0;
  }

  unsigned int doesNotConverge = 1;

  // loop until convergence
  while(doesNotConverge && index < maxIter)
  {
    doesNotConverge = 0;

    for (int i = 0; i < numCent; i++)
    {
      newCentInfo[i].count = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      hostConFlagArr[i] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(devConFlagArr[i],
                hostConFlagPtrArr[i], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);
    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                         devPointData[i],devOldCentSum[i],
                                         devNewCentSum[i],devOldCentCount[i],
                                         devNewCentCount[i],numPnts[i],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                             devCentData[i],devOldCentData[i],
                                             devOldCentSum[i],devNewCentSum[i],
                                             devMaxDriftArr[i],devOldCentCount[i],
                                             devNewCentCount[i],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaMemcpy(allCentInfo[i],
                            devCentInfo[i], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaMemcpy(allCentData[i],
                            devCentData[i], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[i],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[i],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[i],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      assignPointsSimple<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[i],
                                                           devCentInfo[i],
                                                           devPointData[i],
                                                           devPointLwrs[i],
                                                           devCentData[i],
                                                           devMaxDriftArr[i],
                                                           numPnts[i],numCent,
                                                           numGrp,numDim);
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[i],
                                           devConFlagArr[i],
                                           numPnts[i]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[i],
          devConFlagArr[i], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < numGPU; i++)
    {
      if (hostConFlagArr[i])
      {
        doesNotConverge = 1;
      }
    }
  }

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                        devPointData[i],devOldCentSum[i],
                                        devNewCentSum[i],devOldCentCount[i],
                                        devNewCentCount[i],numPnts[i],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                             devCentData[i],devOldCentData[i],
                                             devOldCentSum[i],devNewCentSum[i],
                                             devMaxDriftArr[i],devOldCentCount[i],
                                             devNewCentCount[i],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(allCentInfo[i],
                          devCentInfo[i], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(allCentData[i],
                          devCentData[i], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
        gpuErrchk(cudaSetDevice(i));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[i],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
        gpuErrchk(cudaSetDevice(i));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[i],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  cudaDeviceSynchronize();

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((i*numPnt/numGPU)),
                devPointInfo[i], sizeof(PointInfo)*numPnts[i], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (int i = 0; i < numGPU; i++)
  {
    cudaFree(devPointInfo[i]);
    cudaFree(devPointData[i]);
    cudaFree(devPointLwrs[i]);
    cudaFree(devCentInfo[i]);
    cudaFree(devCentData[i]);
    cudaFree(devMaxDriftArr[i]);
    cudaFree(devNewCentSum[i]);
    cudaFree(devOldCentSum[i]);
    cudaFree(devNewCentCount[i]);
    cudaFree(devOldCentCount[i]);
    cudaFree(devConFlagArr[i]);
  }

  free(allCentInfo);
  free(allCentData);
  free(newCentInfo);
  free(newCentData);
  free(oldCentData);
  free(pointLwrs);

  endTime = omp_get_wtime();
  return endTime - startTime;
}

void calcWeightedMeans(CentInfo *newCentInfo,
                       CentInfo **allCentInfo,
                       DTYPE *newCentData,
                       DTYPE *oldCentData,
                       DTYPE **allCentData,
                       DTYPE *newMaxDriftArr,
                       const int numCent,
                       const int numGrp,
                       const int numDim,
                       const int numGPU)
{
  DTYPE numerator = 0;
  DTYPE denominator = 0;
  DTYPE zeroNumerator = 0;
  int zeroCount = 0;

  for (int i = 0; i < numCent; i++)
  {
      for (int j = 0; j < numDim; j++)
      {
          oldCentData[(i * numDim) + j] = newCentData[(i * numDim) + j];
      }
  }

  for (int i = 0; i < numGPU; i++)
  {
      for (int j = 0; j < numCent; j++)
      {
        newCentInfo[j].count += allCentInfo[i][j].count;

        newCentInfo[j].groupNum = allCentInfo[0][j].groupNum;
      }
  }

  for (int j = 0; j < numCent; j++)
  {
      for (int k = 0; k < numDim; k++)
      {
          for (int l = 0; l < numGPU; l++)
          {
              if (allCentInfo[l][j].count == 0)
              {
                  zeroCount++;
                  zeroNumerator += allCentData[l][(j * numDim) + k];
              }

              numerator +=
              allCentData[l][(j * numDim) + k]*allCentInfo[l][j].count;

              denominator += allCentInfo[l][j].count;
          }

          if (denominator != 0)
          {
              newCentData[(j * numDim) + k] = numerator/denominator;
          }

          else
          {
              newCentData[(j * numDim) + k] = zeroNumerator/zeroCount;
          }

          zeroCount = 0;
          zeroNumerator = 0;
          numerator = 0;
          denominator = 0;
      }

      newCentInfo[j].drift = calcDisCPU(&newCentData[j*numDim],
                                           &oldCentData[j*numDim],
                                           numDim);

      if (newCentInfo[j].drift > newMaxDriftArr[newCentInfo[j].groupNum])
        {
          newMaxDriftArr[newCentInfo[j].groupNum] = newCentInfo[j].drift;
        }
  }
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
  cudaMalloc(&devMaxDrift, sizeof(DTYPE));

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);
  
  DTYPE *devOldCentData = NULL;
  cudaMalloc(&devOldCentData, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  cudaMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),cudaMemcpyHostToDevice));


  clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift, 1);

  // do single run of naive kmeans for initial centroid assignments
  initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
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

    gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                           sizeof(unsigned int), cudaMemcpyHostToDevice));

    // clear maintained data on device
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift, 1);

    // calculate data necessary to make new centroids
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDrift,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

    assignPointsSuper<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                              devPointData,devPointLwrs,
                                              devCentData,devMaxDrift,
                                              numPnt,numCent,1,numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
                                         devConFlag,
                                         numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr, devConFlag,
                         sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDrift,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt, cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData,
                         sizeof(DTYPE)*numDim*numCent, cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devPointLwrs);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devMaxDrift);
  cudaFree(devNewCentSum);
  cudaFree(devOldCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devOldCentCount);
  cudaFree(devConFlag);

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
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                         sizeof(unsigned int), cudaMemcpyHostToDevice));


  clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                                 devNewCentCount,
                                                 numCent,
                                                 numDim);


  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;

    gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),cudaMemcpyHostToDevice));

    assignPointsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                              devCentInfo,
                                              devPointData,
                                              devCentData,
                                              numPnt,
                                              numCent,
                                              numDim);

    clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                                   devNewCentCount,
                                                   numCent,
                                                   numDim);
    // calculate data necessary to make new centroids
    calcCentDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                              devPointData,
                                              devNewCentSum,
                                              devNewCentCount,
                                              numPnt,
                                              numDim);

    // make new centroids
    calcNewCentroidsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                                  devCentInfo,
                                                  devCentData,
                                                  devNewCentSum,
                                                  devNewCentCount,
                                                  numCent,
                                                  numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr,devConFlag,
                           sizeof(unsigned int),cudaMemcpyDeviceToHost));
  }
  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt,cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData,devCentData,
                         sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devNewCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devConFlag);

  endTime = omp_get_wtime();
  return endTime - startTime;
}

PointInfo *storePointInfoOnGPU(PointInfo *pointInfo,
                               const int numPnt)
{
  PointInfo *devPointInfo = NULL;
  gpuErrchk(cudaMalloc(&devPointInfo, sizeof(PointInfo)*numPnt));
  gpuErrchk(cudaMemcpy(devPointInfo, pointInfo,
                         sizeof(PointInfo)*numPnt, cudaMemcpyHostToDevice));
  return devPointInfo;
}

CentInfo *storeCentInfoOnGPU(CentInfo *centInfo,
                             const int numCent)
{
  CentInfo *devCentInfo = NULL;
  gpuErrchk(cudaMalloc(&devCentInfo, sizeof(CentInfo) * numCent));
  gpuErrchk(cudaMemcpy(devCentInfo, centInfo,
                         sizeof(CentInfo)*numCent, cudaMemcpyHostToDevice));
  return devCentInfo;
}

DTYPE *storeDataOnGPU(DTYPE *data,
                      const int numVec,
                      const int numFeat)
{
  DTYPE *devData = NULL;
  gpuErrchk(cudaMalloc(&devData, sizeof(DTYPE) * numVec * numFeat));
  gpuErrchk(cudaMemcpy(devData, data,
                         sizeof(DTYPE)*numVec*numFeat, cudaMemcpyHostToDevice));
  return devData;
}

void warmupGPU()
{
  unsigned int *dev_tmp;
  unsigned int *tmp;
  tmp = (unsigned int*)malloc(sizeof(unsigned int));
  *tmp = 0;
  cudaMalloc((unsigned int**)&dev_tmp, sizeof(unsigned int));

  warmup<<<1,256>>>(dev_tmp);

  cudaMemcpy(tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(dev_tmp);
}

void warmupGPU(const int numGPU)
{
  for (int i = 0; i < numGPU; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
}


// distcalc overloads

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
                    unsigned long long int *countPtr)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;
  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;

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
  cudaMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentData = NULL;
  cudaMalloc(&devOldCentData, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  cudaMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                        sizeof(unsigned int),cudaMemcpyHostToDevice));


  clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);

  // do single run of naive kmeans for initial centroid assignments
  initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
                                       devCentInfo,
                                       devPointData,
                                       devPointLwrs,
                                       devCentData,
                                       numPnt,
                                       numCent,
                                       numGrp,
                                       numDim,
                                       countPtr);


  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;

    gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                           sizeof(unsigned int),cudaMemcpyHostToDevice));

    // clear maintained data on device
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr, numGrp);

    // calculate data necessary to make new centroids
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

    assignPointsFull<<<NBLOCKS, BLOCKSIZE, grpLclSize>>>(devPointInfo,devCentInfo,
                                                         devPointData,devPointLwrs,
                                                         devCentData,devMaxDriftArr,
                                                         numPnt,numCent,numGrp,
                                                         numDim, countPtr);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr,
        devConFlag, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost));
  }
  // calc final centroids (for matching results with lloyds)
  calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDriftArr,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,sizeof(PointInfo)*numPnt,cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData,devCentData,sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devPointLwrs);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devMaxDriftArr);
  cudaFree(devNewCentSum);
  cudaFree(devOldCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devOldCentCount);
  cudaFree(devConFlag);

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
                      const int numGPU,
                      unsigned int *ranIter,
                      unsigned long long int *countPtr)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  int numPnts[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    if (numPnt % numGPU != 0 && i == numGPU-1)
    {
      numPnts[i] = (numPnt / numGPU) + (numPnt % numGPU);
    }
    else
    {
      numPnts[i] = numPnt / numGPU;
    }
  }
  unsigned long long int hostDistCalc = 0;
  unsigned long long int *hostDistCalcCount = &hostDistCalc;

  unsigned long long int *hostDistCalcCountArr;
  hostDistCalcCountArr=(unsigned long long int *)malloc(sizeof(unsigned long long int)*numGPU);
  unsigned long long int *devDistCalcCountArr[numGPU];

  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMalloc(&devDistCalcCountArr[i], sizeof(unsigned long long int)));
  }

  // variable initialization
  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    hostConFlagArr[i] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    hostConFlagPtrArr[i] = &hostConFlagArr[i];
  }

  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;

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
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];
  DTYPE *devPointLwrs[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    cudaSetDevice(i);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[i], sizeof(PointInfo)*(numPnts[i])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[i],
                         pointInfo+(i*numPnt/numGPU),
                         (numPnts[i])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[i], sizeof(DTYPE) * numPnts[i] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[i],
                         pointData+((i*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[i]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[i], sizeof(DTYPE) * numPnts[i] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[i],
                         pointLwrs+((i*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[i]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[i], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[i], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[i],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[i], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[i],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devMaxDriftArr[i], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentSum[i], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentSum[i], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentCount[i], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentCount[i], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devConFlagArr[i], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[i],
              hostConFlagPtrArr[i], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[i],
                                              devOldCentSum[i],
                                              devNewCentCount[i],
                                              devOldCentCount[i],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[i],
                                         devCentInfo[i],
                                         devPointData[i],
                                         devPointLwrs[i],
                                         devCentData[i],
                                         numPnts[i],
                                         numCent,
                                         numGrp,
                                         numDim,
                                         devDistCalcCountArr[i]);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (int i = 0; i < numGPU; i++)
  {
    allCentInfo[i] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (int i = 0; i < numGPU; i++)
  {
    allCentData[i] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
  }

  CentInfo *newCentInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);

  DTYPE *newCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);
  for (int i = 0; i < numCent; i++)
  {
    for (int j = 0; j < numDim; j++)
    {
      newCentData[(i * numDim) + j] = 0;
    }
  }

  DTYPE *oldCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  DTYPE *newMaxDriftArr;
  newMaxDriftArr=(DTYPE *)malloc(sizeof(DTYPE)*numGrp);
  for (int i = 0; i < numGrp; i++)
  {
    newMaxDriftArr[i] = 0.0;
  }

  unsigned int doesNotConverge = 1;

  // loop until convergence
  while(doesNotConverge && index < maxIter)
  {
    doesNotConverge = 0;

    for (int i = 0; i < numCent; i++)
    {
      newCentInfo[i].count = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      hostConFlagArr[i] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(devConFlagArr[i],
                hostConFlagPtrArr[i], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                         devPointData[i],devOldCentSum[i],
                                         devNewCentSum[i],devOldCentCount[i],
                                         devNewCentCount[i],numPnts[i],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                             devCentData[i],devOldCentData[i],
                                             devOldCentSum[i],devNewCentSum[i],
                                             devMaxDriftArr[i],devOldCentCount[i],
                                             devNewCentCount[i],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaMemcpy(allCentInfo[i],
                            devCentInfo[i], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
        gpuErrchk(cudaSetDevice(i));
        gpuErrchk(cudaMemcpy(allCentData[i],
                            devCentData[i], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[i],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[i],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (int i = 0; i < numGPU; i++)
      {
          gpuErrchk(cudaSetDevice(i));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[i],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      assignPointsSimple<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[i],
                                                           devCentInfo[i],
                                                           devPointData[i],
                                                           devPointLwrs[i],
                                                           devCentData[i],
                                                           devMaxDriftArr[i],
                                                           numPnts[i],numCent,
                                                           numGrp,numDim,
                                                           devDistCalcCountArr[i]);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[i],
                                           devConFlagArr[i],
                                           numPnts[i]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[i],
          devConFlagArr[i], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < numGPU; i++)
    {
      if (hostConFlagArr[i])
      {
        doesNotConverge = 1;
      }
    }
  }
  
  cudaDeviceSynchronize();
  
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    gpuErrchk(cudaMemcpy(&hostDistCalcCountArr[i],
                devDistCalcCountArr[i], sizeof(unsigned long long int),
                            cudaMemcpyDeviceToHost));
  }

  for (int i = 0; i < numGPU; i++)
  {
    //printf("hostDistCalcCountArr[%d]: %llu\n", i, hostDistCalcCountArr[i]);
    *hostDistCalcCount += hostDistCalcCountArr[i];
  }

  //printf("hostDistCalcCount: %llu\n", *hostDistCalcCount);
  
  *countPtr = *hostDistCalcCount;

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                        devPointData[i],devOldCentSum[i],
                                        devNewCentSum[i],devOldCentCount[i],
                                        devNewCentCount[i],numPnts[i],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[i],devCentInfo[i],
                                             devCentData[i],devOldCentData[i],
                                             devOldCentSum[i],devNewCentSum[i],
                                             devMaxDriftArr[i],devOldCentCount[i],
                                             devNewCentCount[i],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(allCentInfo[i],
                          devCentInfo[i], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      gpuErrchk(cudaMemcpy(allCentData[i],
                          devCentData[i], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
        gpuErrchk(cudaSetDevice(i));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[i],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
        gpuErrchk(cudaSetDevice(i));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[i],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  cudaDeviceSynchronize();

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((i*numPnt/numGPU)),
                devPointInfo[i], sizeof(PointInfo)*numPnts[i], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (int i = 0; i < numGPU; i++)
  {
    cudaFree(devPointInfo[i]);
    cudaFree(devPointData[i]);
    cudaFree(devPointLwrs[i]);
    cudaFree(devCentInfo[i]);
    cudaFree(devCentData[i]);
    cudaFree(devMaxDriftArr[i]);
    cudaFree(devNewCentSum[i]);
    cudaFree(devOldCentSum[i]);
    cudaFree(devNewCentCount[i]);
    cudaFree(devOldCentCount[i]);
    cudaFree(devConFlagArr[i]);
  }

  free(allCentInfo);
  free(allCentData);
  free(newCentInfo);
  free(newCentData);
  free(oldCentData);
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
                     unsigned int *ranIter,
                     unsigned long long int *countPtr)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization

  unsigned int hostConFlag = 1;

  unsigned int *hostConFlagPtr = &hostConFlag;

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
  cudaMalloc(&devMaxDrift, sizeof(DTYPE));

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);
  
  DTYPE *devOldCentData = NULL;
  cudaMalloc(&devOldCentData, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devOldCentCount = NULL;
  cudaMalloc(&devOldCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),cudaMemcpyHostToDevice));


  clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                            devOldCentSum,
                                            devNewCentCount,
                                            devOldCentCount,
                                            numCent,
                                            numDim);

  clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift, 1);

  // do single run of naive kmeans for initial centroid assignments
  initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
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

    gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                           sizeof(unsigned int), cudaMemcpyHostToDevice));

    // clear maintained data on device
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift, 1);

    // calculate data necessary to make new centroids
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                         devPointData,devOldCentSum,
                                         devNewCentSum,devOldCentCount,
                                         devNewCentCount,numPnt,numDim);

    // make new centroids
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDrift,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

    assignPointsSuper<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                              devPointData,devPointLwrs,
                                              devCentData,devMaxDrift,
                                              numPnt,numCent,1,numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,
                                         devConFlag,
                                         numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr, devConFlag,
                         sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                       devPointData,devOldCentSum,
                                       devNewCentSum,devOldCentCount,
                                       devNewCentCount,numPnt,numDim);

  // make new centroids
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,devCentInfo,
                                             devCentData,devOldCentData,
                                             devOldCentSum,devNewCentSum,
                                             devMaxDrift,devOldCentCount,
                                             devNewCentCount,numCent,numDim);

  cudaDeviceSynchronize();

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt, cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData,
                         sizeof(DTYPE)*numDim*numCent, cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devPointLwrs);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devMaxDrift);
  cudaFree(devNewCentSum);
  cudaFree(devOldCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devOldCentCount);
  cudaFree(devConFlag);

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
                       unsigned int *ranIter,
                       unsigned long long int *countPtr)
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
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  unsigned int *devNewCentCount = NULL;
  cudaMalloc(&devNewCentCount, sizeof(unsigned int) * numCent);

  unsigned int *devConFlag = NULL;
  cudaMalloc(&devConFlag, sizeof(unsigned int));

  gpuErrchk(cudaMemcpy(devConFlag, hostConFlagPtr,
                         sizeof(unsigned int), cudaMemcpyHostToDevice));


  clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                                 devNewCentCount,
                                                 numCent,
                                                 numDim);


  // loop until convergence
  while(hostConFlag && index < maxIter)
  {
    hostConFlag = 0;

    gpuErrchk(cudaMemcpy(devConFlag,hostConFlagPtr,
                         sizeof(unsigned int),cudaMemcpyHostToDevice));

    assignPointsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                              devCentInfo,
                                              devPointData,
                                              devCentData,
                                              numPnt,
                                              numCent,
                                              numDim);

    clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
                                                   devNewCentCount,
                                                   numCent,
                                                   numDim);
    // calculate data necessary to make new centroids
    calcCentDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                              devPointData,
                                              devNewCentSum,
                                              devNewCentCount,
                                              numPnt,
                                              numDim);

    // make new centroids
    calcNewCentroidsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo,
                                                  devCentInfo,
                                                  devCentData,
                                                  devNewCentSum,
                                                  devNewCentCount,
                                                  numCent,
                                                  numDim);

    checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo,devConFlag,numPnt);
    index++;
    gpuErrchk(cudaMemcpy(hostConFlagPtr,devConFlag,
                           sizeof(unsigned int),cudaMemcpyDeviceToHost));
  }
  cudaDeviceSynchronize();
  *countPtr = (unsigned long long int)numPnt * 
    (unsigned long long int)numCent * (unsigned long long int)index;

  // only need the point info for assignments
  gpuErrchk(cudaMemcpy(pointInfo, devPointInfo,
                         sizeof(PointInfo)*numPnt,cudaMemcpyDeviceToHost));
  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData,devCentData,
                         sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  cudaFree(devPointInfo);
  cudaFree(devPointData);
  cudaFree(devCentInfo);
  cudaFree(devCentData);
  cudaFree(devNewCentSum);
  cudaFree(devNewCentCount);
  cudaFree(devConFlag);

  endTime = omp_get_wtime();
  return endTime - startTime;
}