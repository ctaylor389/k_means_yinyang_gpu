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
                    const int numGPU,
                    unsigned int *ranIter)
{
  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
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
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         numGrp,
                                         numDim);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsFull<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDriftArr[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           numGrp,numDim);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDriftArr[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
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
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         numGrp,
                                         numDim);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsSimple<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDriftArr[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           numGrp,numDim);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDriftArr[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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
                     const int numGPU,
                     unsigned int *ranIter)
{
  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

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
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];
  DTYPE *devPointLwrs[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter]));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU)),
                         sizeof(DTYPE)*numPnts[gpuIter],
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDrift[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDrift[gpuIter], sizeof(DTYPE));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentData[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift[gpuIter], 1);
  }

  // do single run of naive kmeans for initial centroid assignments
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         1,
                                         numDim);
  }


  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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
  newMaxDriftArr=(DTYPE *)malloc(sizeof(DTYPE)*1);
  for (int i = 0; i < 1; i++)
  {
    newMaxDriftArr[i] = 0.0;
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                  newCentInfo, sizeof(cent)*numCent,
                              cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                  newCentData, sizeof(DTYPE)*numCent*numDim,
                              cudaMemcpyHostToDevice));
    }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift[gpuIter], 1);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDrift[gpuIter],
                                             devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],
                                             numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < 1; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(devMaxDrift[gpuIter],
                      newMaxDriftArr, sizeof(DTYPE),
                                cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsSuper<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDrift[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           1,numDim);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDrift[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);
    cudaDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDrift[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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

double startLloydOnGPU(PointInfo *pointInfo,
                      CentInfo *centInfo,
                      DTYPE *pointData,
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numDim,
                      const int maxIter,
                      const int numGPU,
                      unsigned int *ranIter)
{
  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

  int index = 1;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);

  // store dataset on device
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  unsigned int doesNotConverge = 1;

  // loop until convergence
  while(doesNotConverge && index < maxIter)
  {
    doesNotConverge = 0;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsLloyd<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                               devCentInfo[gpuIter],
                                               devPointData[gpuIter],
                                               devCentData[gpuIter],
                                               numPnts[gpuIter],
                                               numCent,
                                               numDim);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numCent,numDim);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroidsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],
                                             devCentInfo[gpuIter],
                                             devCentData[gpuIter],
                                             devNewCentSum[gpuIter],
                                             devNewCentCount[gpuIter],
                                             numCent,numDim);

    }

    if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      for (int i = 0; i < numCent; i++)
      {
        newCentInfo[i].count = 0;
      }

      calcWeightedMeansLloyd(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, numCent, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);
    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
  }

  free(allCentInfo);
  free(allCentData);
  free(newCentInfo);
  free(newCentData);
  free(oldCentData);

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

void warmupGPU(const int numGPU)
{
  int gpuIter;
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);
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
                      const int numGPU,
                      unsigned int *ranIter,
                      unsigned long long int *countPtr)
{
  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

  unsigned long long int hostDistCalc = 0;
  unsigned long long int *hostDistCalcCount = &hostDistCalc;

  unsigned long long int *hostDistCalcCountArr;
  hostDistCalcCountArr=(unsigned long long int *)malloc(sizeof(unsigned long long int)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostDistCalcCountArr[gpuIter] = 0;
  }
  unsigned long long int *devDistCalcCountArr[numGPU];


  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(cudaMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), cudaMemcpyHostToDevice));
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
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         numGrp,
                                         numDim,
                                         devDistCalcCountArr[gpuIter]);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsFull<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDriftArr[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           numGrp,numDim,
                                                           devDistCalcCountArr[gpuIter]);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  *hostDistCalcCount = 0;

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));


  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            cudaMemcpyDeviceToHost));
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    *hostDistCalcCount += hostDistCalcCountArr[gpuIter];
  }

  *countPtr = *hostDistCalcCount;

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDriftArr[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

  unsigned long long int hostDistCalc = 0;
  unsigned long long int *hostDistCalcCount = &hostDistCalc;

  unsigned long long int *hostDistCalcCountArr;
  hostDistCalcCountArr=(unsigned long long int *)malloc(sizeof(unsigned long long int)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostDistCalcCountArr[gpuIter] = 0;
  }
  unsigned long long int *devDistCalcCountArr[numGPU];


  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(cudaMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), cudaMemcpyHostToDevice));
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
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU) * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         numGrp,
                                         numDim,
                                         devDistCalcCountArr[gpuIter]);
  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < numGrp; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));
          gpuErrchk(cudaMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsSimple<<<NBLOCKS,BLOCKSIZE,grpLclSize>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDriftArr[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           numGrp,numDim,
                                                           devDistCalcCountArr[gpuIter]);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  *hostDistCalcCount = 0;

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDriftArr[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));


  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            cudaMemcpyDeviceToHost));
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    *hostDistCalcCount += hostDistCalcCountArr[gpuIter];
  }

  *countPtr = *hostDistCalcCount;

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDriftArr[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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
                     const int numGPU,
                     unsigned int *ranIter,
                     unsigned long long int *countPtr)
{

  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

  unsigned long long int hostDistCalc = 0;
  unsigned long long int *hostDistCalcCount = &hostDistCalc;

  unsigned long long int *hostDistCalcCountArr;
  hostDistCalcCountArr=(unsigned long long int *)malloc(sizeof(unsigned long long int)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostDistCalcCountArr[gpuIter] = 0;
  }
  unsigned long long int *devDistCalcCountArr[numGPU];


  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(cudaMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  }

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
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];
  DTYPE *devPointLwrs[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter]));

    gpuErrchk(cudaMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+((gpuIter*numPnt/numGPU)),
                         sizeof(DTYPE)*numPnts[gpuIter],
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  DTYPE *devMaxDrift[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devMaxDrift[gpuIter], sizeof(DTYPE));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentData[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift[gpuIter], 1);
  }

  // do single run of naive kmeans for initial centroid assignments
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devCentInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devPointLwrs[gpuIter],
                                         devCentData[gpuIter],
                                         numPnts[gpuIter],
                                         numCent,
                                         1,
                                         numDim,
                                         devDistCalcCountArr[gpuIter]);
  }


  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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
  newMaxDriftArr=(DTYPE *)malloc(sizeof(DTYPE)*1);
  for (int i = 0; i < 1; i++)
  {
    newMaxDriftArr[i] = 0.0;
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                  newCentInfo, sizeof(cent)*numCent,
                              cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                  newCentData, sizeof(DTYPE)*numCent*numDim,
                              cudaMemcpyHostToDevice));
    }
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
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDrift[gpuIter], 1);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDrift[gpuIter],
                                             devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],
                                             numCent,numDim);

    }

    if (numGPU > 1)
    {
      for (int i = 0; i < 1; i++)
      {
        newMaxDriftArr[i] = 0.0;
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(devMaxDrift[gpuIter],
                      newMaxDriftArr, sizeof(DTYPE),
                                cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsSuper<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                                           devCentInfo[gpuIter],
                                                           devPointData[gpuIter],
                                                           devPointLwrs[gpuIter],
                                                           devCentData[gpuIter],
                                                           devMaxDrift[gpuIter],
                                                           numPnts[gpuIter],numCent,
                                                           1,numDim, devDistCalcCountArr[gpuIter]);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  // calculate data necessary to make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],devCentInfo[gpuIter],
                                             devCentData[gpuIter],devOldCentData[gpuIter],
                                             devOldCentSum[gpuIter],devNewCentSum[gpuIter],
                                             devMaxDrift[gpuIter],devOldCentCount[gpuIter],
                                             devNewCentCount[gpuIter],numCent,numDim);
  }

  if (numGPU > 1)
  {
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          cudaMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          cudaMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(cudaSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                cudaMemcpyHostToDevice));
    }
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);
    cudaDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    gpuErrchk(cudaMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            cudaMemcpyDeviceToHost));
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    *hostDistCalcCount += hostDistCalcCountArr[gpuIter];
  }

  *countPtr = *hostDistCalcCount;

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devPointLwrs[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devMaxDrift[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devOldCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devOldCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
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

double startLloydOnGPU(PointInfo *pointInfo,
                      CentInfo *centInfo,
                      DTYPE *pointData,
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numDim,
                      const int maxIter,
                      const int numGPU,
                      unsigned int *ranIter,
                      unsigned long long int *countPtr)
{
  // start timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // variable initialization
  int gpuIter;

  int numPnts[numGPU];
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    if (numPnt % numGPU != 0 && gpuIter == numGPU-1)
    {
      numPnts[gpuIter] = (numPnt / numGPU) + (numPnt % numGPU);
    }

    else
    {
      numPnts[gpuIter] = numPnt / numGPU;
    }
  }

  unsigned int hostConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagArr[gpuIter] = 1;
  }

  unsigned int *hostConFlagPtrArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hostConFlagPtrArr[gpuIter] = &hostConFlagArr[gpuIter];
  }

  int index = 1;

  unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);

  // store dataset on device
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(cudaMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devPointInfo[gpuIter],
                         pointInfo+(gpuIter*numPnt/numGPU),
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(cudaMemcpy(devPointData[gpuIter],
                         pointData+((gpuIter*numPnt/numGPU) * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         cudaMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(cudaMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        cudaMemcpyHostToDevice));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              cudaMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  CentInfo **allCentInfo = (CentInfo **)malloc(sizeof(CentInfo*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentInfo[gpuIter] = (CentInfo *)malloc(sizeof(CentInfo)*numCent);
  }

  DTYPE **allCentData = (DTYPE **)malloc(sizeof(DTYPE*)*numGPU);
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    allCentData[gpuIter] = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
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

  unsigned int doesNotConverge = 1;

  // loop until convergence
  while(doesNotConverge && index < maxIter)
  {
    doesNotConverge = 0;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      hostConFlagArr[gpuIter] = 0;
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                cudaMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      assignPointsLloyd<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                               devCentInfo[gpuIter],
                                               devPointData[gpuIter],
                                               devCentData[gpuIter],
                                               numPnts[gpuIter],
                                               numCent,
                                               numDim);

    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numCent,numDim);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcCentDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      calcNewCentroidsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devPointInfo[gpuIter],
                                             devCentInfo[gpuIter],
                                             devCentData[gpuIter],
                                             devNewCentSum[gpuIter],
                                             devNewCentCount[gpuIter],
                                             numCent,numDim);

    }

    if (numGPU > 1)
    {
      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            cudaMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(cudaSetDevice(gpuIter));
        gpuErrchk(cudaMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            cudaMemcpyDeviceToHost));
      }

      for (int i = 0; i < numCent; i++)
      {
        newCentInfo[i].count = 0;
      }

      calcWeightedMeansLloyd(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, numCent, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  cudaMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(cudaSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(cudaMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  cudaMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);
    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(cudaSetDevice(gpuIter));
      gpuErrchk(cudaMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      cudaMemcpyDeviceToHost));
    }

    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      if (hostConFlagArr[gpuIter])
      {
        doesNotConverge = 1;
      }
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));
    cudaDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(cudaSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(cudaMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], cudaMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(cudaMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,cudaMemcpyDeviceToHost));

  *countPtr = numPnt * numCent * index;

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    cudaFree(devPointInfo[gpuIter]);
    cudaFree(devPointData[gpuIter]);
    cudaFree(devCentInfo[gpuIter]);
    cudaFree(devCentData[gpuIter]);
    cudaFree(devNewCentSum[gpuIter]);
    cudaFree(devNewCentCount[gpuIter]);
    cudaFree(devConFlagArr[gpuIter]);
  }

  free(allCentInfo);
  free(allCentData);
  free(newCentInfo);
  free(newCentData);
  free(oldCentData);

  endTime = omp_get_wtime();
  return endTime - startTime;
}