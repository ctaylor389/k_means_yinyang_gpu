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
  cudaMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

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
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDriftArr,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);

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
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDriftArr,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
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
  cudaMalloc(&devMaxDriftArr, sizeof(DTYPE) * numGrp);

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

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
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDriftArr,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);
    
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
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDriftArr,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
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
    printf("    **Assigned %d points to GPU %d\n", numPnt, i);
    
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
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];
  DTYPE *devPointLwrs[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    cudaSetDevice(i);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[i], sizeof(PointInfo)*(numPnts[i])));
    printf("    **Allocated %d PointInfo Objects on GPU %d\n", numPnts[i], i);

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[i],
                         pointInfo+(i*numPnt/numGPU),
                         (numPnts[i])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));
    printf("    **Copied %d PointInfo Objects onto GPU %d\n", numPnts[i], i);

    gpuErrchk(cudaMalloc(&devPointData[i], sizeof(DTYPE) * numPnts[i] * numDim));
    printf("    **Allocated %d DTYPE values (point data) on GPU %d\n", 
           numPnts[i] * numDim, i);

    // possible error here ???
    gpuErrchk(cudaMemcpy(devPointData[i],
                         pointData+((i*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[i]*numDim,
                         cudaMemcpyHostToDevice));
    printf("    **Copied %d DTYPE values (point data) onto GPU %d\n", 
           numPnts[i] * numDim, i);

    gpuErrchk(cudaMalloc(&devPointLwrs[i], sizeof(DTYPE) * numPnts[i] *
                         numGrp));
    printf("    **Allocated %d DTYPE values (lwr bounds) on GPU %d\n", 
           numPnts[i] * numGrp, i);

    gpuErrchk(cudaMemcpy(devPointLwrs[i], 
                         pointLwrs+((i*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[i]*numGrp,
                         cudaMemcpyHostToDevice));
    printf("    **Copied %d DTYPE values (lwr bounds) onto GPU %d\n", 
           numPnts[i] * numGrp, i);
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[i], sizeof(CentInfo)*numCent));
    printf("    **Allocated %d CentInfo Objects on GPU %d\n", numCent, i);

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[i],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));
    printf("    **Copied %d CentInfo Objects onto GPU %d\n", numCent, i);

    gpuErrchk(cudaMalloc(&devCentData[i], sizeof(DTYPE)*numCent*numDim));
    printf("    **Allocated %d DTYPE values (centroid data) on GPU %d\n", 
           numCent * numDim, i);
    gpuErrchk(cudaMemcpy(devCentData[i],
                        centInfo, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
    printf("    **Copied %d DTYPE values (centroid data) onto GPU %d\n", 
           numPnts[i] * numDim, i);
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devMaxDriftArr[i], sizeof(DTYPE) * numGrp);
    printf("    **Allocated %d DTYPE values (max drift array) on GPU %d\n", 
           numGrp, i);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentSum[i], sizeof(DTYPE) * numCent * numDim);
    printf("    **Allocated %d DTYPE values (new centroid sums) on GPU %d\n", 
           numCent * numDim, i);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentSum[i], sizeof(DTYPE) * numCent * numDim);
    printf("    **Allocated %d DTYPE values (old centroid sums) on GPU %d\n", 
           numCent * numDim, i);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devNewCentCount[i], sizeof(unsigned int) * numCent);
    printf("    **Allocated %d unsigned int values (new centroid count) on GPU %d\n", numCent, i);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    cudaMalloc(&devOldCentCount[i], sizeof(unsigned int) * numCent);
    printf("    **Allocated %d unsigned int values (old centroid count) on GPU %d\n", numCent, i);
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
    printf("    **Running clearCentCalcData on GPU %d\n", i);
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
    printf("    **Running clearDriftArr on GPU %d\n", i);
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);
    
  }

  #pragma omp parallel for num_threads(numGPU)
  for (int i = 0; i < numGPU; i++)
  {
    gpuErrchk(cudaSetDevice(i));
    printf("    **Running initRunKernel on GPU %d\n", i);
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



  unsigned int doesNotConverge = 1;

  // loop until convergence
  while(doesNotConverge && index < maxIter)
  {
    doesNotConverge = 0;

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
      printf("    **Running clearDriftArr on GPU %d\n", i);
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[i], numGrp);
      
    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      printf("    **Running calcCentData on GPU %d\n", i);
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
      printf("    **Running calcNewCentroids on GPU %d\n", i);
      calcNewCentroids<<<NBLOCKS,BLOCKSIZE,oldPosSize>>>(devPointInfo[i],
                                                         devCentInfo[i],
                                                         devCentData[i],
                                                         devOldCentSum[i],
                                                         devNewCentSum[i],
                                                         devMaxDriftArr[i],
                                                         devOldCentCount[i],
                                                         devNewCentCount[i],
                                                         numCent,numDim);
      
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      cudaSetDevice(i);
      printf("    **GPU %d is waiting at the barrier\n", i);
      cudaDeviceSynchronize();
    }

    #pragma omp parallel for num_threads(numGPU)
    for (int i = 0; i < numGPU; i++)
    {
      gpuErrchk(cudaSetDevice(i));
      printf("    **Running assignPointsSimple on GPU %d\n", i);
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
      printf("  **Running checkConverge on GPU %d\n", i);
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
    printf("    **Running final calcCentData on GPU %d\n", i);
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
    printf("    **Running final calcNewCent on GPU %d\n", i);
    calcNewCentroids<<<NBLOCKS,BLOCKSIZE,oldPosSize>>>(devPointInfo[i],
                                                       devCentInfo[i],
                                                       devCentData[i],
                                                       devOldCentSum[i],
                                                       devNewCentSum[i],
                                                       devMaxDriftArr[i],
                                                       devOldCentCount[i],
                                                       devNewCentCount[i],
                                                       numCent,numDim);
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
  gpuErrchk(cudaMemcpy(centData,devCentData[0],
                       sizeof(DTYPE)*numDim*numCent,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (int i = 0; i < numGPU; i++)
  {
    printf("    **Freeing data on GPU %d\n", i);
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
  cudaMalloc(&devMaxDrift, sizeof(DTYPE));

  // centroid calculation data
  DTYPE *devNewCentSum = NULL;
  cudaMalloc(&devNewCentSum, sizeof(DTYPE) * numCent * numDim);

  DTYPE *devOldCentSum = NULL;
  cudaMalloc(&devOldCentSum, sizeof(DTYPE) * numCent * numDim);

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
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                         devCentData,devOldCentSum,
                                                         devNewCentSum,devMaxDrift,
                                                         devOldCentCount,devNewCentCount,
                                                         numCent,numDim);
    
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
  calcNewCentroids<<<NBLOCKS, BLOCKSIZE, oldPosSize>>>(devPointInfo,devCentInfo,
                                                       devCentData,devOldCentSum,
                                                       devNewCentSum,devMaxDrift,
                                                       devOldCentCount,devNewCentCount,
                                                       numCent,numDim);
  
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
