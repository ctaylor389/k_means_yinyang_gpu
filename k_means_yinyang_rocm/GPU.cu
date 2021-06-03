#include "hip/hip_runtime.h"
#include "GPU.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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



point *storeDatasetOnGPU(point *dataset,
                         unsigned long int numPnt)
{
    point * dev_inputData = NULL;

    // alloc dataset to GPU
    gpuErrchk(hipMalloc(&dev_inputData, numPnt*sizeof(point)));

    // copy input data to GPU
    gpuErrchk(hipMemcpy(dev_inputData, 
                dataset, numPnt*sizeof(point), 
                            hipMemcpyHostToDevice));

    unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);
    return dev_inputData;

}

cent *storeCentDataOnGPU(cent *centDataset,
                         const unsigned int numCent)
{
    cent * dev_centInputData = NULL;


    // alloc dataset and drift array to GPU
    gpuErrchk(hipMalloc(&dev_centInputData, numCent*sizeof(cent)));
    

    // copy input data to GPU
    gpuErrchk(hipMemcpy(dev_centInputData, 
                centDataset, numCent*sizeof(cent), 
                            hipMemcpyHostToDevice));


    unsigned int NBLOCKS = ceil(numCent*1.0/BLOCKSIZE*1.0);
    
    return dev_centInputData;
}



/////////////////////////////////////////////
// Host functions for calling CUDA kernels //
/////////////////////////////////////////////

int startFullOnGPU(point *hostDataset,
                   cent *hostCentDataset,
                   unsigned long long int *hostDistCalcCount,
                   double *fullStartTime,
                   double *fullEndTime,
                   unsigned int *ranIter)
{
    int countFlag;

    if(hostDistCalcCount == NULL)
    countFlag = 0;
    else
    countFlag = 1;

    // start timer
    *fullStartTime = omp_get_wtime();

    // variable initialization

    unsigned int hostConFlag = 1;

    unsigned int *hostConFlagPtr = &hostConFlag;
    
    unsigned int index = 0;

    unsigned int NBLOCKS = ceil(NPOINT*1.0/BLOCKSIZE*1.0);


    // group centroids
    groupCent(hostCentDataset, NCLUST, NGROUP, NDIM);

    // store dataset on device
    point *devDataset;

    devDataset = storeDatasetOnGPU(hostDataset, NPOINT);

    // store centroids on device
    cent *devCentDataset;

    devCentDataset = storeCentDataOnGPU(hostCentDataset, NCLUST);

    unsigned long long int *devDistCalcCount = NULL;

    if(countFlag)
    {
        // allocate device-only data
        gpuErrchk(hipMalloc(&devDistCalcCount, sizeof(unsigned long long int)));

        gpuErrchk(hipMemcpy(devDistCalcCount, 
                hostDistCalcCount, sizeof(unsigned long long int), 
                            hipMemcpyHostToDevice));
    }

    DTYPE *devMaxDriftArr = NULL;
    hipMalloc(&devMaxDriftArr, NGROUP*sizeof(DTYPE));

    // centroid calculation data
    struct vector *devNewCentSum = NULL;
    hipMalloc(&devNewCentSum, NCLUST*sizeof(vector));

    struct vector *devOldCentSum = NULL;
    hipMalloc(&devOldCentSum, NCLUST*sizeof(vector));

    unsigned int *devNewCentCount = NULL;
    hipMalloc(&devNewCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devOldCentCount = NULL;
    hipMalloc(&devOldCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devConFlag = NULL;
    hipMalloc(&devConFlag, sizeof(unsigned int));

    gpuErrchk(hipMemcpy(devConFlag,
            hostConFlagPtr, sizeof(unsigned int),
                        hipMemcpyHostToDevice));


    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                              devOldCentSum,
                                              devNewCentCount,
                                              devOldCentCount);

    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);

    // do single run of naive kmeans for initial centroid assignments
    if(countFlag)
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset, devDistCalcCount);
    else
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset);

    // loop until convergence
    while(hostConFlag && index < MAXITER)
    {
        hostConFlag = 0;
        
        gpuErrchk(hipMemcpy(devConFlag,
            hostConFlagPtr, sizeof(unsigned int),
                        hipMemcpyHostToDevice));

        // clear maintained data on device
        hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);

        hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                  devOldCentSum,
                                                  devNewCentCount,
                                                  devOldCentCount);

        // calculate data necessary to make new centroids
        hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                             devCentDataset,
                                             devOldCentSum,
                                             devNewCentSum,
                                             devOldCentCount,
                                             devNewCentCount);

        // make new centroids
        hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                 devCentDataset,
                                                 devMaxDriftArr,
                                                 devOldCentSum,
                                                 devNewCentSum,
                                                 devOldCentCount,
                                                 devNewCentCount);

        if(countFlag)
        {
            hipLaunchKernelGGL(assignPointsFull, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                     devCentDataset,
                                                     devMaxDriftArr,
                                                     devDistCalcCount);
        }
        else
        {
            hipLaunchKernelGGL(assignPointsFull, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                     devCentDataset,
                                                     devMaxDriftArr);
        }

        hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devConFlag);
        index++;
        gpuErrchk(hipMemcpy(hostConFlagPtr,
            devConFlag, sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    }
    hipDeviceSynchronize();

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(hostDataset,
                devDataset, NPOINT*sizeof(struct point),
                            hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(hostCentDataset,
                devCentDataset, NCLUST*sizeof(struct cent),
                            hipMemcpyDeviceToHost));

    *fullEndTime = omp_get_wtime();

    if(countFlag)
    {
        gpuErrchk(hipMemcpy(hostDistCalcCount, 
                   devDistCalcCount, sizeof(unsigned long long int), 
                                hipMemcpyDeviceToHost));
        hipFree(devDistCalcCount);
    }


    *ranIter = index + 1;


    // clean up, return
    hipFree(devMaxDriftArr);
    hipFree(devNewCentSum);
    hipFree(devOldCentSum);
    hipFree(devNewCentCount);
    hipFree(devOldCentCount);
    hipFree(devDataset);
    hipFree(devCentDataset);
    hipFree(devConFlag);

    
    return 0;
}

/*
function containing master loop that calls yinyang kernels
*/
int startSimpleOnGPU(point *hostDataset,
                     cent *hostCentDataset,
                     unsigned long long int *hostDistCalcCount,
                     double *simpStartTime,
                     double *simpEndTime,
                     unsigned int *ranIter)
{
    unsigned int countFlag;
    if(hostDistCalcCount == NULL)
    countFlag = 0;
    else
    countFlag = 1;

    // start timer
    *simpStartTime = omp_get_wtime();

    // variable initialization

    unsigned int hostConFlag = 1;
    unsigned int *hostConFlagPtr = &hostConFlag;
    unsigned int index = 0;
    unsigned int NBLOCKS = ceil(NPOINT*1.0/BLOCKSIZE*1.0);

    // group centroids
    groupCent(hostCentDataset, NCLUST, NGROUP, NDIM);

    // store dataset on device
    point *devDataset;

    devDataset = storeDatasetOnGPU(hostDataset, NPOINT);

    // store centroids on device
    cent *devCentDataset;

    devCentDataset = storeCentDataOnGPU(hostCentDataset,
                                        NCLUST);

    // allocate device-only data
    unsigned long long int *devDistCalcCount = NULL;

    if(countFlag)
    {
        // allocate device-only data
        gpuErrchk(hipMalloc(&devDistCalcCount, sizeof(unsigned long long int)));

        gpuErrchk(hipMemcpy(devDistCalcCount, 
                hostDistCalcCount, sizeof(unsigned long long int), 
                            hipMemcpyHostToDevice));
    }


    DTYPE *devMaxDriftArr = NULL;
    hipMalloc(&devMaxDriftArr, NGROUP*sizeof(DTYPE));

    // centroid calculation data
    struct vector *devNewCentSum = NULL;
    hipMalloc(&devNewCentSum, NCLUST*sizeof(vector));

    struct vector *devOldCentSum = NULL;
    hipMalloc(&devOldCentSum, NCLUST*sizeof(vector));

    unsigned int *devNewCentCount = NULL;
    hipMalloc(&devNewCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devOldCentCount = NULL;
    hipMalloc(&devOldCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devConFlag = NULL;
    hipMalloc(&devConFlag, sizeof(unsigned int));

    gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                        hipMemcpyHostToDevice));

    

    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                              devOldCentSum,
                                              devNewCentCount,
                                              devOldCentCount);
                                              
    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);



    // do single run of naive kmeans for initial centroid assignments
    if(countFlag)
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset, devDistCalcCount);
    else
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset);

    // loop until convergence
    while(hostConFlag && index < MAXITER)
    {
        hostConFlag = 0;
        
        gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                        hipMemcpyHostToDevice));	

        // clear maintained data on device
        hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);
        
        /*hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                  devOldCentSum,
                                                  devNewCentCount,
                                                  devOldCentCount);*/


        // calculate data necessary to make new centroids
        hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                              devCentDataset,
                                              devOldCentSum,
                                              devNewCentSum,
                                              devOldCentCount,
                                              devNewCentCount);

        // make new centroids
        hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                  devCentDataset,
                                                  devMaxDriftArr,
                                                  devOldCentSum,
                                                  devNewCentSum,
                                                  devOldCentCount,
                                                  devNewCentCount);

        hipDeviceSynchronize();
        // update point assignments via assignPointsernel
        if(countFlag)
        {
            hipLaunchKernelGGL(assignPointsSimple, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                       devCentDataset,
                                                       devMaxDriftArr,
                                                       devDistCalcCount);
        }
        else
        {
            hipLaunchKernelGGL(assignPointsSimple, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                       devCentDataset,
                                                       devMaxDriftArr);
        }

        hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devConFlag);
        index++;
        gpuErrchk(hipMemcpy(hostConFlagPtr, 
            devConFlag, sizeof(unsigned int), 
                        hipMemcpyDeviceToHost));
    }
    hipDeviceSynchronize();

    // copy finished clusters and points from device to host
    gpuErrchk(hipMemcpy(hostDataset,
                devDataset, NPOINT*sizeof(struct point),
                            hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(hostCentDataset,
                devCentDataset, NCLUST*sizeof(struct cent),
                            hipMemcpyDeviceToHost));

    *simpEndTime = omp_get_wtime();

    if(countFlag)
    {
        gpuErrchk(hipMemcpy(hostDistCalcCount, 
                    devDistCalcCount, sizeof(unsigned long long int), 
                                hipMemcpyDeviceToHost));
        hipFree(devDistCalcCount);
    }

    *ranIter = index + 1;

    // clean up, return
    hipFree(devMaxDriftArr);
    hipFree(devNewCentSum);
    hipFree(devOldCentSum);
    hipFree(devNewCentCount);
    hipFree(devOldCentCount);
    hipFree(devDataset);
    hipFree(devCentDataset);
    hipFree(devConFlag);
    
    return 0;
}




int startLloydOnGPU(point *hostDataset,
                    cent *hostCentDataset,
                    double *lloydStartTime,
                    double *lloydEndTime,
                    unsigned int *ranIter)
{


    // start timer
    *lloydStartTime = omp_get_wtime();

    unsigned int hostConFlag = 1;
    unsigned int *hostConFlagPtr = &hostConFlag;
    unsigned int index = 0;

    // store dataset on device
    point *devDataset;

    devDataset = storeDatasetOnGPU(hostDataset, NPOINT);

    // store centroids on device
    cent *devCentDataset;

    devCentDataset = storeCentDataOnGPU(hostCentDataset,
                                        NCLUST);

    unsigned int NBLOCKS = ceil(NPOINT*1.0/BLOCKSIZE*1.0);

    unsigned int *devNewCentCount = NULL;
    hipMalloc(&devNewCentCount, NCLUST * sizeof(unsigned int));

    vector *devNewCentSum = NULL;
    hipMalloc(&devNewCentSum, NCLUST * sizeof(vector));

    unsigned int *devConFlag = NULL;
    hipMalloc(&devConFlag, sizeof(unsigned int));

    gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                        hipMemcpyHostToDevice));
    

    hipLaunchKernelGGL(clearCentCalcDataLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                    devNewCentCount);

    // master loop for maxIter runs
    while(hostConFlag && index < MAXITER)
    {
        hostConFlag = 0;
        
        gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                        hipMemcpyHostToDevice));
        
        // update point assignments via assignPointsernel
        hipLaunchKernelGGL(assignPointsLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                   devCentDataset);


        hipLaunchKernelGGL(calcCentDataLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                              devCentDataset,
                                              devNewCentSum,
                                              devNewCentCount);
    
        hipLaunchKernelGGL(calcNewCentroidsLloyd, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                  devCentDataset,
                                                  devNewCentSum,
                                                  devNewCentCount);

        hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devConFlag);					   
        
        index++;
        gpuErrchk(hipMemcpy(hostConFlagPtr, 
            devConFlag, sizeof(unsigned int), 
                        hipMemcpyDeviceToHost));
    }
    hipDeviceSynchronize();
    
    // copy assigned data from device to host
    gpuErrchk(hipMemcpy(hostDataset,
                devDataset, NPOINT * sizeof(point),
                            hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(hostCentDataset,
                devCentDataset, NCLUST * sizeof(cent),
                            hipMemcpyDeviceToHost));
    *lloydEndTime = omp_get_wtime();

    *ranIter = index;

    hipFree(devDataset);
    hipFree(devCentDataset);
    hipFree(devNewCentCount);
    hipFree(devNewCentSum);
    hipFree(devConFlag);
    return 0;
}



/*
function containing master loop that 
calls yinyang super simplified kernels
*/
int startSuperOnGPU(point *hostDataset,
                      cent *hostCentDataset,
                      unsigned long long int *hostDistCalcCount,
                      double *supStartTime,
                      double *supEndTime,
                      unsigned int *ranIter)
{

    unsigned int countFlag;
    if(hostDistCalcCount == NULL)
    countFlag = 0;
    else
    countFlag = 1;

    // start timer
    *supStartTime = omp_get_wtime();

    unsigned int clustIndex;
    unsigned int index = 0;
    unsigned int hostConFlag = 1;
    unsigned int *hostConFlagPtr = &hostConFlag;
    unsigned int NBLOCKS = ceil(NPOINT*1.0/BLOCKSIZE*1.0);

    // assign all centroids to 0 group
    for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
    {
        hostCentDataset[clustIndex].groupNum = 0;
    }

    // store dataset on device
    point *devDataset;

    devDataset = storeDatasetOnGPU(hostDataset, NPOINT);

    // store centroids on device
    cent *devCentDataset;

    devCentDataset = storeCentDataOnGPU(hostCentDataset,
                                        NCLUST);

    // allocate a count on the GPU
    unsigned long long int *devDistCalcCount = NULL;
    
    if(countFlag)
    {
        gpuErrchk(hipMalloc(&devDistCalcCount, sizeof(unsigned long long int)));

        gpuErrchk(hipMemcpy(devDistCalcCount, 
                hostDistCalcCount, sizeof(unsigned long long int), 
                            hipMemcpyHostToDevice));
    }

    DTYPE *devMaxDriftArr = NULL;
    hipMalloc(&devMaxDriftArr, NGROUP*sizeof(DTYPE));

    // centroid calculation data
    struct vector *devNewCentSum = NULL;
    hipMalloc(&devNewCentSum, NCLUST*sizeof(vector));

    struct vector *devOldCentSum = NULL;
    hipMalloc(&devOldCentSum, NCLUST*sizeof(vector));

    unsigned int *devNewCentCount = NULL;
    hipMalloc(&devNewCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devOldCentCount = NULL;
    hipMalloc(&devOldCentCount, NCLUST*sizeof(unsigned int));

    unsigned int *devConFlag = NULL;
    hipMalloc(&devConFlag, sizeof(unsigned int));

    gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                        hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                  devOldCentSum,
                                                  devNewCentCount,
                                                  devOldCentCount);

    hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);

    // do single run of naive kmeans for initial centroid assignments	
    if(countFlag)
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset, devDistCalcCount);
    else
    hipLaunchKernelGGL(initRunKernel, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devCentDataset);

    // master loop for maxIter runs
    while(hostConFlag && index < MAXITER)
    {
        hostConFlag = 0;

        gpuErrchk(hipMemcpy(devConFlag, 
            hostConFlagPtr, sizeof(unsigned int), 
                            hipMemcpyHostToDevice));

        hipLaunchKernelGGL(clearDriftArr, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devMaxDriftArr);

        hipLaunchKernelGGL(clearCentCalcData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devNewCentSum,
                                                  devOldCentSum,
                                                  devNewCentCount,
                                                  devOldCentCount);

        hipLaunchKernelGGL(calcCentData, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                              devCentDataset,
                                              devOldCentSum,
                                              devNewCentSum,
                                              devOldCentCount,
                                              devNewCentCount);
    
        hipLaunchKernelGGL(calcNewCentroids, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                  devCentDataset,
                                                  devMaxDriftArr,
                                                  devOldCentSum,
                                                  devNewCentSum,
                                                  devOldCentCount,
                                                  devNewCentCount);

        // update point assignments via assignPointsernel
        if(countFlag)
        {
            hipLaunchKernelGGL(assignPointsSuper, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                      devCentDataset,
                                                      devMaxDriftArr,
                                                      devDistCalcCount);
        }
        else
        {
            hipLaunchKernelGGL(assignPointsSuper, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset,
                                                      devCentDataset,
                                                      devMaxDriftArr);            
        }

        hipLaunchKernelGGL(checkConverge, dim3(NBLOCKS), dim3(BLOCKSIZE), 0, 0, devDataset, devConFlag);

        index++;
        gpuErrchk(hipMemcpy(hostConFlagPtr, 
            devConFlag, sizeof(unsigned int), 
                        hipMemcpyDeviceToHost));
    }
    hipDeviceSynchronize();

    // copy assigned data from device to host
    gpuErrchk(hipMemcpy(hostDataset,
                devDataset, NPOINT*sizeof(point),
                            hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(hostCentDataset,
                devCentDataset, NCLUST*sizeof(cent),
                            hipMemcpyDeviceToHost));

    *supEndTime = omp_get_wtime();

    if(countFlag)
    {
        gpuErrchk(hipMemcpy(hostDistCalcCount, 
                    devDistCalcCount, sizeof(unsigned long long int), 
                                hipMemcpyDeviceToHost));
        hipFree(devDistCalcCount);
    }


    *ranIter = index + 1;

    hipFree(devDataset);
    hipFree(devCentDataset);
    hipFree(devMaxDriftArr);
    hipFree(devNewCentSum);
    hipFree(devOldCentSum);
    hipFree(devNewCentCount);
    hipFree(devOldCentCount);
    hipFree(devConFlag);
    
    return 0;
}



