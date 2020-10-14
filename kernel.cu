#include "kernel.h"


__global__ void warmup(unsigned int * tmp)
{
	if(threadIdx.x == 0)
	{
		*tmp = 555;
	}
	return;
}

__global__ void checkConverge(struct point * dataset,
							  unsigned int * conFlag)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

	if(tid >= NPOINT)
	{
		return;
	}

	if(dataset[tid].oldCentroid != dataset[tid].centroidIndex)
	{	
		atomicCAS(conFlag, 0, 1);
	}
	
}


/*
global kernel that assigns data points to their first
centroid assignment. Runs exactly once.
*/
__global__ void initRunKernel(struct point * dataset,
							  struct cent * centroidDataset,
							  unsigned long long int *calcCount)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);


	if(tid >= NPOINT)
	{
		return;
	}


	unsigned int centIndex;
	double currDistance;
	double currMin = INFINITY;
	unsigned int centGroup;

	
	for(centIndex = 0; centIndex < NCLUST; centIndex++)
	{
		// calculate euclidean distance between point and centroid
		currDistance = calcDis(&dataset[tid].vec, 
							   &centroidDataset[centIndex].vec);
		//atomicAdd(calcCount, 1);
		if(currDistance < currMin)
		{

			// make the former current min the new 
			// lower bound for it's group
			if(currMin != INFINITY)
			{
				centGroup = 
					centroidDataset[dataset[tid].centroidIndex].groupNum;
				dataset[tid].lwrBoundArr[centGroup] = currMin;
				
			}
			// update assignment and upper bound
			dataset[tid].centroidIndex = centIndex;
			dataset[tid].uprBound = currDistance;
			

			//update currMin
			currMin = currDistance;
			
		}
		else
		{
			centGroup = centroidDataset[centIndex].groupNum;
			if(currDistance < dataset[tid].lwrBoundArr[centGroup])
			{
				dataset[tid].lwrBoundArr[centGroup] = currDistance;
			}
		}

	}
	currMin = INFINITY;


}





/*
simple helper kernel that clears the drift array of size T
on the GPU. Called once each iteration for a total of MAXITER times
*/
__global__ void clearDriftArr(double *devMaxDriftArr)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

	if(tid >= NGROUP)
	{
		return;
	}
	devMaxDriftArr[tid] = 0.0;
}

__global__ void clearCentCalcData(struct vector *newCentSum,
							  	  struct vector *oldCentSum,
							  	  unsigned int *newCentCount,
							  	  unsigned int *oldCentCount)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

	if(tid >= NCLUST)
	{
		return;
	}

	unsigned int dimIndex;
	
	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{
		newCentSum[tid].feat[dimIndex] = 0.0;
		oldCentSum[tid].feat[dimIndex] = 0.0;
	}
	newCentCount[tid] = 0;
	oldCentCount[tid] = 0;
	
}

__global__ void clearCentCalcDataLloyd(struct vector *newCentSum,
							  	  	   unsigned int *newCentCount)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

	if(tid >= NCLUST)
	{
		return;
	}

	unsigned int dimIndex;
	
	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{
		newCentSum[tid].feat[dimIndex] = 0.0;
	}
	newCentCount[tid] = 0;
}

/*
device helper function provides a double precision implementation 
of atomicMax using atomicCAS 
*/
__device__ void atomicMax(double *const address, const double value)
{
	if (*address >= value)
	{
		return;
	}

	unsigned long long int * const address_as_i = (unsigned long long int *)address;
	unsigned long long int old = * address_as_i, assumed;


	do
	{
		assumed = old;
		if(__longlong_as_double(assumed) >= value)
		{
			break;
		}
		old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
	}while(assumed != old);

	
}



/*
Lloyd's algorithm implementation of point assignment kernel.
Assigns all points in the dataset to a centroid based on new
centroid data.
*/
__global__ void assignPointsLloyd(struct point * dataset,
						struct cent * centroidDataset)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NPOINT)
	{
		return;
	}

	dataset[tid].oldCentroid = dataset[tid].centroidIndex;

	double currMin = INFINITY;
	double currDis;
	unsigned int index;

	for(index = 0; index < NCLUST; index++)
	{
		currDis = calcDis(&dataset[tid].vec, 
						  &centroidDataset[index].vec);
		if(currDis < currMin)
		{
			dataset[tid].centroidIndex = index;
			currMin = currDis;
		}
	}
}

// A shared memory implementation of Yinyang's point assignment step
__global__ void assignPointsYinyang(struct point *dataset,
								  struct cent *centroidDataset,
								  double *maxDriftArr,
								  unsigned long long int *calcCount)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NPOINT)
	{
		return;
	}

	// variable declaration
	unsigned int btid = threadIdx.x;
	dataset[tid].oldCentroid = dataset[tid].centroidIndex;
	double tmpGlobLwr = INFINITY;
	unsigned int index;

	__shared__ unsigned int groupLclArr[NGROUP*BLOCKSIZE];

	// update points upper bound
	dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;


	
	// update group lower bounds
	// for all in lower bound array
	for(index = 0; index < NGROUP; index++)
	{
		
		// subtract lowerbound by group's drift
		dataset[tid].lwrBoundArr[index] -= maxDriftArr[index];

		// if the lowerbound is less than the temp global lower,
		if(dataset[tid].lwrBoundArr[index] < tmpGlobLwr)
		{
			// lower bound is new temp global lower
			tmpGlobLwr = dataset[tid].lwrBoundArr[index];
		}
	}



	
	// if the global lower bound is less than the upper bound
	if(tmpGlobLwr < dataset[tid].uprBound)
	{

		// tighten upper bound ub = d(x, b(x))
		dataset[tid].uprBound =
			calcDis(&dataset[tid].vec,
				&centroidDataset[dataset[tid].centroidIndex].vec);

		//atomicAdd(calcCount, 1);


		// if the lower bound is less than the upper bound
		if(tmpGlobLwr < dataset[tid].uprBound)
		{
			// loop through groups
			for(index = 0; index < NGROUP; index++)
			{
				// if the lower bound is less than the upper bound
				if(dataset[tid].lwrBoundArr[index] < dataset[tid].uprBound)
				{
					// that lower bounds group is marked
					groupLclArr[index + (btid * NGROUP)] = 1;
				}
				else
				{
					groupLclArr[index + (btid * NGROUP)] = 0;
				}
			}
			// execute point calcs given the groups
			pointCalcs(&dataset[tid],
					   centroidDataset,
					   &groupLclArr[btid * NGROUP],
					   maxDriftArr,
					   calcCount);
		}

	}
}



__global__ void assignPointsHamerly(struct point *dataset,
									struct cent *centroidDataset,
									double *maxDrift,
									unsigned long long int *calcCount)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NPOINT)
	{
		return;
	}

	// set centroid's old centroid to be current assignment
	dataset[tid].oldCentroid = dataset[tid].centroidIndex;

	// point calc variables
	unsigned int clustIndex;
	double compDistance;


	// update bounds
	dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;
	dataset[tid].lwrBoundArr[0] -= *maxDrift;


	if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
	{
		// tighten upper bound
		dataset[tid].uprBound =
			calcDis(&dataset[tid].vec,
					&centroidDataset[dataset[tid].centroidIndex].vec);

		//atomicAdd(calcCount, 1);


		if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
		{
			// to get a new lower bound
			dataset[tid].lwrBoundArr[0] = INFINITY;

			
			for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
			{
				// do not calculate for the already assigned cluster
				if(clustIndex == dataset[tid].centroidIndex)
				{
					continue;
				}

				compDistance = calcDis(&dataset[tid].vec,
									   &centroidDataset[clustIndex].vec);
				//atomicAdd(calcCount, 1);
				
				if(compDistance < dataset[tid].uprBound)
				{
					dataset[tid].lwrBoundArr[0] = dataset[tid].uprBound;
					dataset[tid].centroidIndex = clustIndex;
					dataset[tid].uprBound = compDistance;
				}
				else if(compDistance < dataset[tid].lwrBoundArr[0])
				{
					dataset[tid].lwrBoundArr[0] = compDistance;
				}
			}
		}
	}
}


__device__ void pointCalcs(struct point *pointPtr,
						   struct cent *centroidDataset,
						   unsigned int *groupArr,
						   double *maxDriftArr,
						   unsigned long long int *calcCount)
{

	unsigned int index;
	double compDistance;
	double minDistance = pointPtr->uprBound;


	for(index = 0; index < NGROUP; index++)
	{
		if(groupArr[index])
		{
			pointPtr->lwrBoundArr[index] = INFINITY;
		}
		else if(pointPtr->lwrBoundArr[index] != INFINITY)
		{
			pointPtr->lwrBoundArr[index] -= maxDriftArr[index];
		}

	}



	for(index = 0; index < NCLUST; index++)
	{

		if(groupArr[centroidDataset[index].groupNum])
		{
			if(index == pointPtr->centroidIndex)
			{
				continue;
			}
			compDistance = calcDis(&pointPtr->vec, 
									&centroidDataset[index].vec);
			
			//atomicAdd(calcCount, 1);
			
			if(compDistance < minDistance)
			{
				pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = minDistance;
				minDistance = compDistance;
				pointPtr->centroidIndex = index;
				pointPtr->uprBound = compDistance;
			}
			else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[index].groupNum])
			{
				pointPtr->lwrBoundArr[centroidDataset[index].groupNum] = compDistance;
			}
		}
	}
}





__global__ void calcCentData(struct point *dataset,
						struct cent *centroidDataset,
						struct vector *oldSums,
						struct vector *newSums,
						unsigned int *oldCounts,
						unsigned int *newCounts)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NPOINT)
	{
		return;
	}

	unsigned int dimIndex;

	// atomicAdd 1 to old and new counts corresponding
	if(dataset[tid].oldCentroid >= 0)
	atomicAdd(&oldCounts[dataset[tid].oldCentroid], 1);
	
	atomicAdd(&newCounts[dataset[tid].centroidIndex], 1);

	// if old assignment and new assignment are not equal
	if(dataset[tid].oldCentroid != dataset[tid].centroidIndex)
	{
		// for all values in the vector
		for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
		{
			// atomic add the point's vector to the sum count
			if(dataset[tid].oldCentroid >= 0)
			{
				atomicAdd(&oldSums[dataset[tid].oldCentroid].feat[dimIndex],
					dataset[tid].vec.feat[dimIndex]);
			}	
			atomicAdd(&newSums[dataset[tid].centroidIndex].feat[dimIndex],
				dataset[tid].vec.feat[dimIndex]);
		}
	}
	
}




__global__ void calcNewCentroids(struct point *dataset,
						struct cent *centroidDataset,
						double *maxDriftArr,
						struct vector *oldSums,
						struct vector *newSums,
						unsigned int *oldCounts,
						unsigned int *newCounts)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

	if(tid >= NCLUST)
	{
		return;
	}

	double oldFeature;
	double oldSumFeat;
	double newSumFeat;
	double compDrift;

	
	unsigned int dimIndex;


	struct vector oldVec;

	// create the new centroid vector
	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{
		if(newCounts[tid] > 0)
		{
			oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
			oldFeature = centroidDataset[tid].vec.feat[dimIndex];
			oldSumFeat = oldSums[tid].feat[dimIndex];
			newSumFeat = newSums[tid].feat[dimIndex];

			centroidDataset[tid].vec.feat[dimIndex] = 
				(oldFeature * oldCounts[tid] - oldSumFeat + newSumFeat)
															/ newCounts[tid];			
							
			newSums[tid].feat[dimIndex] = 0.0;
			oldSums[tid].feat[dimIndex] = 0.0;
		}
		else
		{
			oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];

			// no change to centroid

			newSums[tid].feat[dimIndex] = 0.0;
			oldSums[tid].feat[dimIndex] = 0.0;
		}
	}
	

	// calculate the centroid's drift
	compDrift = calcDis(&oldVec, &centroidDataset[tid].vec);

	atomicMax(&maxDriftArr[centroidDataset[tid].groupNum], compDrift);

	// set the centroid's vector to the new vector
	centroidDataset[tid].drift = compDrift;

	// clear the count and the sum arrays
	
	oldCounts[tid] = 0;
	newCounts[tid] = 0;
	
}


__global__ void calcCentDataLloyd(struct point *dataset,
						struct cent *centroidDataset,
						struct vector *newSums,
						unsigned int *newCounts)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NPOINT)
	{
		return;
	}

	unsigned int dimIndex;

	// atomicAdd 1 to new counts corresponding
	atomicAdd(&newCounts[dataset[tid].centroidIndex], 1);

	// for all values in the vector
	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{		
		atomicAdd(&newSums[dataset[tid].centroidIndex].feat[dimIndex],
			dataset[tid].vec.feat[dimIndex]);
	}
	
}

__global__ void calcNewCentroidsLloyd(struct point *dataset,
						struct cent *centroidDataset,
						struct vector *newSums,
						unsigned int *newCounts)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NCLUST)
	{
		return;
	}
	unsigned int dimIndex;


	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{
		if(newCounts[tid] > 0)
		{
			centroidDataset[tid].vec.feat[dimIndex] =
				newSums[tid].feat[dimIndex] / newCounts[tid];
		}
		// otherwise, no change
		newSums[tid].feat[dimIndex] = 0.0;
	}


	newCounts[tid] = 0;
}
__global__ void calcNewCentroidsAve(struct point *dataset,
						struct cent *centroidDataset,
						struct vector *newSums,
						unsigned int *newCounts,
						double *maxDriftArr)
{
	unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
	if(tid >= NCLUST)
	{
		return;
	}
	unsigned int dimIndex;

	double compDrift;

	struct vector oldVec; 

	for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
	{
		if(newCounts[tid] > 0)
		{
			oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
		
			centroidDataset[tid].vec.feat[dimIndex] =
				newSums[tid].feat[dimIndex] / newCounts[tid];

			newSums[tid].feat[dimIndex] = 0.0;
		}
		else
		{
			oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
		
			centroidDataset[tid].vec.feat[dimIndex] = 0.0;

			newSums[tid].feat[dimIndex] = 0.0;
		}

	}

	// compute drift
	compDrift = calcDis(&oldVec, &centroidDataset[tid].vec);
	centroidDataset[tid].drift = compDrift;

	atomicMax(&maxDriftArr[centroidDataset[tid].groupNum], compDrift);

	newCounts[tid] = 0;
}



/*
Simple device helper function that takes in two vectors and returns
the euclidean distance between them at double precision
*/
__device__ double calcDis(struct vector *vec1, 
						  struct vector *vec2)
{
	unsigned int index;
	double total = 0;
	double square;

	for(index = 0; index < NDIM; index++)
	{
		square = (vec1->feat[index] - vec2->feat[index]);
		total += square * square;
	}

	return sqrt(total);
}





