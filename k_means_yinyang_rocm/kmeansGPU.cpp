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

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+(currentAllocations[gpuIter] * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));
          gpuErrchk(hipMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsFull, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDriftArr[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+(currentAllocations[gpuIter] * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));
          gpuErrchk(hipMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsSimple, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDriftArr[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter]));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+currentAllocations[gpuIter],
                         sizeof(DTYPE)*numPnts[gpuIter],
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDrift[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDrift[gpuIter], sizeof(DTYPE));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentData[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDrift[gpuIter], 1);
  }

  // do single run of naive kmeans for initial centroid assignments
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                  newCentInfo, sizeof(cent)*numCent,
                              hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(hipMemcpy(devCentData[gpuIter],
                  newCentData, sizeof(DTYPE)*numCent*numDim,
                              hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDrift[gpuIter], 1);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(devMaxDrift[gpuIter],
                      newMaxDriftArr, sizeof(DTYPE),
                                hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsSuper, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);
    hipDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDrift[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

  // store dataset on device
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numCent,numDim);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroidsLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
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
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);
    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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

void warmupGPU(const int numGPU)
{
  int gpuIter;
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);
    hipDeviceSynchronize();
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
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(hipMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), hipMemcpyHostToDevice));
  }

  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;

  int index = 1;

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+(currentAllocations[gpuIter] * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));
          gpuErrchk(hipMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsFull, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));


  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            hipMemcpyDeviceToHost));
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
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDriftArr[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(hipMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), hipMemcpyHostToDevice));
  }

  int grpLclSize = sizeof(unsigned int)*numGrp*BLOCKSIZE;

  int index = 1;

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }


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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] *
                         numGrp));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+(currentAllocations[gpuIter] * numGrp),
                         sizeof(DTYPE)*numPnts[gpuIter]*numGrp,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];
  DTYPE *devOldCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));
    
    // alloc the old position data structure
    gpuErrchk(hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numDim * numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDriftArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDriftArr[gpuIter], sizeof(DTYPE) * numGrp);
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    // do single run of naive kmeans for initial centroid assignments
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr[gpuIter], numGrp);

    }


    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));
          gpuErrchk(hipMemcpy(devMaxDriftArr[gpuIter],
                       newMaxDriftArr, numGrp*sizeof(DTYPE),
                                  hipMemcpyHostToDevice));
      }
    }

    /*
    if (numGPU == 2)
    {
      if (index == 20)
      {
        writeData(newCentData, numCent, numDim, "centroidsAt1_2gpu.txt");
      }
    }

    if (numGPU == 3)
    {
      if (index == 20)
      {
        writeData(newCentData, numCent, numDim, "centroidsAt1_3gpu.txt");
      }
    }
    */

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsSimple, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), grpLclSize, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, numGrp, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }  

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));


  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            hipMemcpyDeviceToHost));
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
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDriftArr[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMalloc(&devDistCalcCountArr[gpuIter], sizeof(unsigned long long int)));
    gpuErrchk(hipMemcpy(devDistCalcCountArr[gpuIter], &hostDistCalcCountArr[gpuIter], 
                         sizeof(unsigned long long int), hipMemcpyHostToDevice));
  }

  int index = 1;

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }


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

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointLwrs[gpuIter], sizeof(DTYPE) * numPnts[gpuIter]));

    gpuErrchk(hipMemcpy(devPointLwrs[gpuIter],
                         pointLwrs+currentAllocations[gpuIter],
                         sizeof(DTYPE)*numPnts[gpuIter],
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  DTYPE *devMaxDrift[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devMaxDrift[gpuIter], sizeof(DTYPE));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  DTYPE *devOldCentData[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentData[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devOldCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devOldCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                              devOldCentSum[gpuIter],
                                              devNewCentCount[gpuIter],
                                              devOldCentCount[gpuIter],
                                              numCent,
                                              numDim);

  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDrift[gpuIter], 1);
  }

  // do single run of naive kmeans for initial centroid assignments
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                  newCentInfo, sizeof(cent)*numCent,
                              hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));

      // copy input data to GPU
      gpuErrchk(hipMemcpy(devCentData[gpuIter],
                  newCentData, sizeof(DTYPE)*numCent*numDim,
                              hipMemcpyHostToDevice));
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    // clear maintained data on device
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devMaxDrift[gpuIter], 1);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                         devPointData[gpuIter],devOldCentSum[gpuIter],
                                         devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                         devNewCentCount[gpuIter],numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
      }

      calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
        allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(devMaxDrift[gpuIter],
                      newMaxDriftArr, sizeof(DTYPE),
                                hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsSuper, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);

    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
                                        devPointData[gpuIter],devOldCentSum[gpuIter],
                                        devNewCentSum[gpuIter],devOldCentCount[gpuIter],
                                        devNewCentCount[gpuIter],numPnts[gpuIter],numDim);
  }

  // make new centroids
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],devCentInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                          devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                          hipMemcpyDeviceToHost));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(allCentData[gpuIter],
                          devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                          hipMemcpyDeviceToHost));
    }

    calcWeightedMeans(newCentInfo, allCentInfo, newCentData, oldCentData,
      allCentData, newMaxDriftArr, numCent, 1, numDim, numGPU);

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                    newCentInfo, sizeof(cent)*numCent,
                                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
        gpuErrchk(hipSetDevice(gpuIter));

        // copy input data to GPU
        gpuErrchk(hipMemcpy(devCentData[gpuIter],
                    newCentData, sizeof(DTYPE)*numCent*numDim,
                                hipMemcpyHostToDevice));
    }
  }

  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);
    hipDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    gpuErrchk(hipMemcpy(&hostDistCalcCountArr[gpuIter],
                devDistCalcCountArr[gpuIter], sizeof(unsigned long long int),
                            hipMemcpyDeviceToHost));
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
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devPointLwrs[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devMaxDrift[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devOldCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devOldCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
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

  unsigned int NBLOCKS[numGPU]; 
  for (int i = 0; i < numGPU; i++)
  {
    NBLOCKS[i] = ceil(numPnts[i]*1.0/BLOCKSIZE*1.0);
  }

  // store dataset on device
  PointInfo *devPointInfo[numGPU];
  DTYPE *devPointData[numGPU];

  int currentAllocations[numGPU];
  for (int i = 0; i < numGPU; i++)
  {
    currentAllocations[i] = 0;
    for (int j = 0; j < i; j++)
    {
      currentAllocations[i] += numPnts[j];
    }
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipSetDevice(gpuIter);

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&devPointInfo[gpuIter], sizeof(PointInfo)*(numPnts[gpuIter])));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devPointInfo[gpuIter],
                         pointInfo+currentAllocations[gpuIter],
                         (numPnts[gpuIter])*sizeof(PointInfo),
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devPointData[gpuIter], sizeof(DTYPE) * numPnts[gpuIter] * numDim));

    gpuErrchk(hipMemcpy(devPointData[gpuIter],
                         pointData+(currentAllocations[gpuIter] * numDim),
                         sizeof(DTYPE)*numPnts[gpuIter]*numDim,
                         hipMemcpyHostToDevice));
  }

  // store centroids on device
  CentInfo *devCentInfo[numGPU];
  DTYPE *devCentData[numGPU];

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&devCentInfo[gpuIter], sizeof(CentInfo)*numCent));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                         centInfo, sizeof(CentInfo)*numCent,
                         hipMemcpyHostToDevice));

    gpuErrchk(hipMalloc(&devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim));
    gpuErrchk(hipMemcpy(devCentData[gpuIter],
                        centData, sizeof(DTYPE)*numCent*numDim,
                        hipMemcpyHostToDevice));
  }

  // centroid calculation data
  DTYPE *devNewCentSum[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentSum[gpuIter], sizeof(DTYPE) * numCent * numDim);
  }

  unsigned int *devNewCentCount[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devNewCentCount[gpuIter], sizeof(unsigned int) * numCent);
  }

  unsigned int *devConFlagArr[numGPU];
  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipMalloc(&devConFlagArr[gpuIter], sizeof(unsigned int));
    gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
              hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
              hipMemcpyHostToDevice));
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));
    hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(devConFlagArr[gpuIter],
                hostConFlagPtrArr[gpuIter], sizeof(unsigned int),
                hipMemcpyHostToDevice));
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(assignPointsLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numCent,numDim);

    }

    // calculate data necessary to make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcCentDataLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                         devPointData[gpuIter],
                                         devNewCentSum[gpuIter],
                                         devNewCentCount[gpuIter],
                                         numPnts[gpuIter],numDim);

    }

    // make new centroids
    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(calcNewCentroidsLloyd, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
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
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentInfo[gpuIter],
                            devCentInfo[gpuIter], sizeof(CentInfo)*numCent,
                            hipMemcpyDeviceToHost));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
        gpuErrchk(hipSetDevice(gpuIter));
        gpuErrchk(hipMemcpy(allCentData[gpuIter],
                            devCentData[gpuIter], sizeof(DTYPE)*numCent*numDim,
                            hipMemcpyDeviceToHost));
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
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentInfo[gpuIter],
                      newCentInfo, sizeof(cent)*numCent,
                                  hipMemcpyHostToDevice));
      }

      #pragma omp parallel for num_threads(numGPU)
      for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
      {
          gpuErrchk(hipSetDevice(gpuIter));

          // copy input data to GPU
          gpuErrchk(hipMemcpy(devCentData[gpuIter],
                      newCentData, sizeof(DTYPE)*numCent*numDim,
                                  hipMemcpyHostToDevice));
      }
    }

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS[gpuIter]), dim3(BLOCKSIZE), 0, 0, devPointInfo[gpuIter],
                                           devConFlagArr[gpuIter],
                                           numPnts[gpuIter]);
    }

    index++;

    #pragma omp parallel for num_threads(numGPU)
    for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
    {
      gpuErrchk(hipSetDevice(gpuIter));
      gpuErrchk(hipMemcpy(hostConFlagPtrArr[gpuIter],
          devConFlagArr[gpuIter], sizeof(unsigned int),
                      hipMemcpyDeviceToHost));
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
    gpuErrchk(hipSetDevice(gpuIter));
    hipDeviceSynchronize();
  }

  #pragma omp parallel for num_threads(numGPU)
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    gpuErrchk(hipSetDevice(gpuIter));

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(pointInfo+((gpuIter*numPnt/numGPU)),
                devPointInfo[gpuIter], sizeof(PointInfo)*numPnts[gpuIter], hipMemcpyDeviceToHost));
  }

  // and the final centroid positions
  gpuErrchk(hipMemcpy(centData, devCentData[0],
                       sizeof(DTYPE)*numCent*numDim,hipMemcpyDeviceToHost));

  *countPtr = (uint64_t)numPnt * (uint64_t)numCent * (uint64_t)index;

  *ranIter = index;

  // clean up, return
  for (gpuIter = 0; gpuIter < numGPU; gpuIter++)
  {
    hipFree(devPointInfo[gpuIter]);
    hipFree(devPointData[gpuIter]);
    hipFree(devCentInfo[gpuIter]);
    hipFree(devCentData[gpuIter]);
    hipFree(devNewCentSum[gpuIter]);
    hipFree(devNewCentCount[gpuIter]);
    hipFree(devConFlagArr[gpuIter]);
  }

  free(allCentInfo);
  free(allCentData);
  free(newCentInfo);
  free(newCentData);
  free(oldCentData);

  endTime = omp_get_wtime();
  return endTime - startTime;
}