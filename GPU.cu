#include "GPU.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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



point *storeDatasetOnGPU(point *dataset,
						 unsigned long int numPnt)
{
	point * dev_inputData = NULL;

	// alloc dataset to GPU
	gpuErrchk(cudaMalloc(&dev_inputData, numPnt*sizeof(point)));

	// copy input data to GPU
	gpuErrchk(cudaMemcpy(dev_inputData, 
				dataset, numPnt*sizeof(point), 
							cudaMemcpyHostToDevice));

	unsigned int NBLOCKS = ceil(numPnt*1.0/BLOCKSIZE*1.0);
	return dev_inputData;

}

cent *storeCentDataOnGPU(cent *centDataset,
						 const unsigned int numCent)
{
	cent * dev_centInputData = NULL;


	// alloc dataset and drift array to GPU
	gpuErrchk(cudaMalloc(&dev_centInputData, numCent*sizeof(cent)));
	

	// copy input data to GPU
	gpuErrchk(cudaMemcpy(dev_centInputData, 
				centDataset, numCent*sizeof(cent), 
							cudaMemcpyHostToDevice));


	unsigned int NBLOCKS = ceil(numCent*1.0/BLOCKSIZE*1.0);
	
	return dev_centInputData;
}


/*
function containing master loop that calls yinyang kernels
*/
int startYinYangOnGPU(point *hostDataset,
					  cent *hostCentDataset,
					  unsigned long long int *hostDistCalcCount,
					  double *yinStartTime,
					  double *yinEndTime,
					  unsigned int *ranIter)
{

	// start timer
	*yinStartTime = omp_get_wtime();




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

	gpuErrchk(cudaMalloc(&devDistCalcCount, sizeof(unsigned long long int)));
	
	gpuErrchk(cudaMemcpy(devDistCalcCount, 
			hostDistCalcCount, sizeof(unsigned long long int), 
						cudaMemcpyHostToDevice));


	double *devMaxDriftArr = NULL;
	cudaMalloc(&devMaxDriftArr, NGROUP*sizeof(double));

	// centroid calculation data
	struct vector *devNewCentSum = NULL;
	cudaMalloc(&devNewCentSum, NCLUST*sizeof(vector));

	struct vector *devOldCentSum = NULL;
	cudaMalloc(&devOldCentSum, NCLUST*sizeof(vector));

	unsigned int *devNewCentCount = NULL;
	cudaMalloc(&devNewCentCount, NCLUST*sizeof(unsigned int));

	unsigned int *devOldCentCount = NULL;
	cudaMalloc(&devOldCentCount, NCLUST*sizeof(unsigned int));

	unsigned int *devConFlag = NULL;
	cudaMalloc(&devConFlag, sizeof(unsigned int));

	gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));

	

	clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
											  devOldCentSum,
											  devNewCentCount,
											  devOldCentCount);
											  
	clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr);



	// do single run of naive kmeans for initial centroid assignments	
	initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devDataset, 
										 devCentDataset,
										 devDistCalcCount);


	// loop until convergence
	while(hostConFlag && index < MAXITER)
	{
		hostConFlag = 0;
		
		gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));	

						
		// clear maintained data on device
		clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr);
		
		clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
											  	devOldCentSum,
											  	devNewCentCount,
											  	devOldCentCount);


		// calculate data necessary to make new centroids
		calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
					 						 devCentDataset,
					 						 devOldCentSum,
					 						 devNewCentSum,
					 						 devOldCentCount,
					 						 devNewCentCount);

		// make new centroids
		calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
						 						 devCentDataset,
						 						 devMaxDriftArr,
						 						 devOldCentSum,
						 						 devNewCentSum,
						 						 devOldCentCount,
						 						 devNewCentCount);

						 						 
		// update point assignments via assignPointsernel
		assignPointsYinyang<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
												    devCentDataset,
												    devMaxDriftArr,
												    devDistCalcCount);

		checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devDataset, devConFlag);					   
		index++;
		gpuErrchk(cudaMemcpy(hostConFlagPtr, 
			devConFlag, sizeof(unsigned int), 
						cudaMemcpyDeviceToHost));
	}

	cudaDeviceSynchronize();


	//printf("\ntotal iter: %d\n", index);
	
	// copy finished clusters and points from device to host
	gpuErrchk(cudaMemcpy(hostDataset,
				devDataset, NPOINT*sizeof(struct point),
							cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostCentDataset,
				devCentDataset, NCLUST*sizeof(struct cent),
							cudaMemcpyDeviceToHost));

	*yinEndTime = omp_get_wtime();

	*ranIter = index + 1;

	// clean up, return
	cudaFree(devDistCalcCount);
	cudaFree(devMaxDriftArr);
	cudaFree(devNewCentSum);
	cudaFree(devOldCentSum);
	cudaFree(devNewCentCount);
	cudaFree(devOldCentCount);
	cudaFree(devDataset);
	cudaFree(devCentDataset);
	cudaFree(devConFlag);

	
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
	cudaMalloc(&devNewCentCount, NCLUST * sizeof(unsigned int));

	struct vector *devNewCentSum = NULL;
	cudaMalloc(&devNewCentSum, NCLUST * sizeof(vector));



	unsigned int *devConFlag = NULL;
	cudaMalloc(&devConFlag, sizeof(unsigned int));

	gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));
	

	clearCentCalcDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
											 	   devNewCentCount);

	// master loop for maxIter runs
	while(hostConFlag && index < MAXITER)
	{
		hostConFlag = 0;
		
		gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));
		
		// update point assignments via assignPointsernel
		assignPointsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
												   devCentDataset);


		calcCentDataLloyd<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
					 						 devCentDataset,
					 						 devNewCentSum,
					 						 devNewCentCount);
	
		calcNewCentroidsLloyd<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
						 						 devCentDataset,
						 						 devNewCentSum,
						 						 devNewCentCount);


		checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devDataset, devConFlag);					   
		
		index++;
		gpuErrchk(cudaMemcpy(hostConFlagPtr, 
			devConFlag, sizeof(unsigned int), 
						cudaMemcpyDeviceToHost));
	}


	cudaDeviceSynchronize();


	//printf("\ntotal iter: %d\n", index);
	
	// copy assigned data from device to host
	gpuErrchk(cudaMemcpy(hostDataset,
				devDataset, NPOINT * sizeof(point),
							cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostCentDataset,
				devCentDataset, NCLUST * sizeof(cent),
							cudaMemcpyDeviceToHost));
	*lloydEndTime = omp_get_wtime();

	*ranIter = index;

	cudaFree(devDataset);
	cudaFree(devCentDataset);
	cudaFree(devNewCentCount);
	cudaFree(devNewCentSum);
	cudaFree(devConFlag);
	return 0;
}



/*
function containing master loop that calls yinyang kernels
*/
int startHamerlyOnGPU(point *hostDataset,
					  cent *hostCentDataset,
					  unsigned long long int *hostDistCalcCount,
					  double *hamStartTime,
					  double *hamEndTime,
					  unsigned int *ranIter)
{

	// start timer
	*hamStartTime = omp_get_wtime();

	unsigned int hostConFlag = 1;

	unsigned int *hostConFlagPtr = &hostConFlag;
	
	unsigned int index = 0;
	

	unsigned int clustIndex;

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

	gpuErrchk(cudaMalloc(&devDistCalcCount, sizeof(unsigned long long int)));

	gpuErrchk(cudaMemcpy(devDistCalcCount, 
			hostDistCalcCount, sizeof(unsigned long long int), 
						cudaMemcpyHostToDevice));

	double *devMaxDriftArr = NULL;
	cudaMalloc(&devMaxDriftArr, NGROUP*sizeof(double));

	// centroid calculation data
	struct vector *devNewCentSum = NULL;
	cudaMalloc(&devNewCentSum, NCLUST*sizeof(vector));

	struct vector *devOldCentSum = NULL;
	cudaMalloc(&devOldCentSum, NCLUST*sizeof(vector));

	unsigned int *devNewCentCount = NULL;
	cudaMalloc(&devNewCentCount, NCLUST*sizeof(unsigned int));

	unsigned int *devOldCentCount = NULL;
	cudaMalloc(&devOldCentCount, NCLUST*sizeof(unsigned int));


	unsigned int *devConFlag = NULL;
	cudaMalloc(&devConFlag, sizeof(unsigned int));

	gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));
	


	clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
											  	devOldCentSum,
											  	devNewCentCount,
											  	devOldCentCount);



											  
	clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr);

	// do single run of naive kmeans for initial centroid assignments	
	initRunKernel<<<NBLOCKS,BLOCKSIZE>>>(devDataset, 
										 devCentDataset,
										 devDistCalcCount);



	// master loop for maxIter runs
	while(hostConFlag && index < MAXITER)
	{

		hostConFlag = 0;

		gpuErrchk(cudaMemcpy(devConFlag, 
			hostConFlagPtr, sizeof(unsigned int), 
						cudaMemcpyHostToDevice));
	
		clearDriftArr<<<NBLOCKS, BLOCKSIZE>>>(devMaxDriftArr);

		clearCentCalcData<<<NBLOCKS, BLOCKSIZE>>>(devNewCentSum,
												  devOldCentSum,
												  devNewCentCount,
												  devOldCentCount);


		calcCentData<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
					 						 devCentDataset,
					 						 devOldCentSum,
					 						 devNewCentSum,
					 						 devOldCentCount,
					 						 devNewCentCount);
	
		calcNewCentroids<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
						 						 devCentDataset,
						 						 devMaxDriftArr,
						 						 devOldCentSum,
						 						 devNewCentSum,
						 						 devOldCentCount,
						 						 devNewCentCount);

		
		// update point assignments via assignPointsernel
		assignPointsHamerly<<<NBLOCKS, BLOCKSIZE>>>(devDataset,
												    devCentDataset,
												    devMaxDriftArr,
												   	devDistCalcCount);

		checkConverge<<<NBLOCKS,BLOCKSIZE>>>(devDataset, devConFlag);

		index++;
		gpuErrchk(cudaMemcpy(hostConFlagPtr, 
			devConFlag, sizeof(unsigned int), 
						cudaMemcpyDeviceToHost));


	}

	cudaDeviceSynchronize();
	
	

	// copy assigned data from device to host
	gpuErrchk(cudaMemcpy(hostDataset,
				devDataset, NPOINT*sizeof(point),
							cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hostCentDataset,
				devCentDataset, NCLUST*sizeof(cent),
							cudaMemcpyDeviceToHost));


	*hamEndTime = omp_get_wtime();


	*ranIter = index + 1;


	cudaFree(devDataset);
	cudaFree(devCentDataset);
	cudaFree(devMaxDriftArr);
	cudaFree(devNewCentSum);
	cudaFree(devOldCentSum);
	cudaFree(devNewCentCount);
	cudaFree(devOldCentCount);
	cudaFree(devConFlag);
	
	return 0;
}



