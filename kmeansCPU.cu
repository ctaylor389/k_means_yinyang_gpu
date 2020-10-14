#include "kmeansCPU.h"




int startLloydOnCPU(point *dataset,
				 	cent *centroidDataset,
				 	double *startTime,
				 	double *endTime,
				 	unsigned int *ranIter)
{

	*startTime = omp_get_wtime();
	
	unsigned int pntIndex, clustIndex, dimIndex;

	unsigned int index = 0;

	unsigned int conFlag = 0;


	omp_set_num_threads(NTHREAD);

	double currMin, currDis;

	// start standard kmeans algorithm for MAXITER iterations
	while(!conFlag && index < MAXITER)
	{
		currMin = INFINITY;
		// point assignment step

		#pragma omp parallel private(pntIndex, clustIndex, currDis, currMin) shared(dataset, centroidDataset)
		{
			#pragma omp for nowait schedule(static)
			for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
			{
				dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;
				for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
				{
					currDis = calcDisCPU(dataset[pntIndex].vec, 
										 centroidDataset[clustIndex].vec,
										 NDIM);
					if(currDis < currMin)
					{
						dataset[pntIndex].centroidIndex = clustIndex;
						currMin = currDis;
					}
				}
				currMin = INFINITY;
			}
		}
		
		// update centroids

		// clear centroids features
		for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
		{
			for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
			{
				centroidDataset[clustIndex].vec.feat[dimIndex] = 0.0;
			}
			centroidDataset[clustIndex].count = 0;
		}
		// sum all assigned point's features
		for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
		{
			centroidDataset[dataset[pntIndex].centroidIndex].count++;

			for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
			{
				
				centroidDataset[dataset[pntIndex].centroidIndex].vec.feat[dimIndex] +=
					dataset[pntIndex].vec.feat[dimIndex];
			}
		}
		// take the average of each feature to get new centroid features
		for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
		{
			for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
			{
				if(centroidDataset[clustIndex].count > 0)
				{
					centroidDataset[clustIndex].vec.feat[dimIndex] /= 
										centroidDataset[clustIndex].count;
				}
				// otherwise, centroid remains the same
			}
		}
		index++;
		conFlag = checkConverge(dataset);
	}

	*endTime = omp_get_wtime();
	*ranIter = index;

	//printf("iterations: %d\n", index);
	return 0;
}


/*
CPU implementation of the Yinyang algorithm
given only a file name with the points and a start and end time
*/
int startYinyangOnCPU(point *dataset,
				  	  cent *centroidDataset,
				  	  unsigned long long int *distCalcCount,
				  	  double *startTime,
				  	  double *endTime,
				  	  unsigned int *ranIter)
{
	*startTime = omp_get_wtime();

	// index variables
	unsigned int pntIndex, grpIndex;

	unsigned int index = 0;

	unsigned int conFlag = 0;

	// array to contain the maximum drift of each group of centroids
	// note: shared amongst all points
	double driftArr[NGROUPCPU];

	// array to contain integer flags which mark which groups need to be checked
	// for a potential new centroid
	// note: unique to each point
	int groupLclArr[NGROUPCPU];

	omp_set_num_threads(NTHREAD);

	// the minimum of all the lower bounds for a single point
	double tmpGlobLwr = INFINITY;
	

	// cluster the centroids int NGROUP groups
	groupCent(centroidDataset, NCLUST, NGROUPCPU, NDIM);


	// clear both arrays
	for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
	{
		groupLclArr[grpIndex] = 0;
		driftArr[grpIndex] = 0.0;
	}

	// run one iteration of standard kmeans for initial centroid assignments
	initPoints(dataset, centroidDataset);

	// master loop
	while(!conFlag && index < MAXITER)
	{
		// clear drift array each new iteration
		for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
		{
			driftArr[grpIndex] = 0.0;
		}
	
		// update centers via optimised update method
		updateCentroids(dataset, centroidDataset, driftArr);

		// filtering done in parallel
		#pragma omp parallel private(pntIndex, grpIndex, tmpGlobLwr, groupLclArr) shared(dataset, centroidDataset, driftArr)
		{
			#pragma omp for schedule(static)
			for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
			{
				// reset old centroid before possibly finding a new one
				dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

				tmpGlobLwr = INFINITY;
				
				// update upper bound
					// ub = ub + centroid's drift
				dataset[pntIndex].uprBound +=
						centroidDataset[dataset[pntIndex].centroidIndex].drift;
				
				// update group lower bounds
					// lb = lb - maxGroupDrift
				for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
				{
					dataset[pntIndex].lwrBoundArr[grpIndex] -= driftArr[grpIndex];

					if(dataset[pntIndex].lwrBoundArr[grpIndex] < tmpGlobLwr)
					{
						// assign temp global lowerBound
						// for all lowerbounds assigned to point
						tmpGlobLwr = dataset[pntIndex].lwrBoundArr[grpIndex];
					}
				}

				// global filtering
				// if global lowerbound >= upper bound
				if(tmpGlobLwr < dataset[pntIndex].uprBound)
				{
					// tighten upperbound ub = d(x, b(x))
					dataset[pntIndex].uprBound = 
						calcDisCPU(dataset[pntIndex].vec, 
							centroidDataset[dataset[pntIndex].centroidIndex].vec,
							NDIM);

					// check condition again
					if(tmpGlobLwr < dataset[pntIndex].uprBound)
					{
						// group filtering
						for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
						{
							// mark groups that need to be checked
							if(dataset[pntIndex].lwrBoundArr[grpIndex] < 
														dataset[pntIndex].uprBound)
							{
								groupLclArr[grpIndex] = 1;
							}
							else
							{
								groupLclArr[grpIndex] = 0;
							}
						}
						// pass group array and point to go execute distance calculations
						pointCalcs(&dataset[pntIndex], groupLclArr, driftArr, 
																centroidDataset);
					}
				}			
				
				for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
				{
					groupLclArr[grpIndex] = 0;
				}
			}
		}
		index++;
		conFlag = checkConverge(dataset);
	}

	
	*endTime = omp_get_wtime();

	*ranIter = index + 1;


	return 0;
}

unsigned int checkConverge(point *dataset)
{
	int index;
	{
		for(index = 0; index < NPOINT; index++)
		{
			if(dataset[index].centroidIndex != dataset[index].oldCentroid)
			{
				return 0;
			}
		}
	}

	return 1;
}


void pointCalcs(point *pointPtr, 
				int *groupArr, 
				double *driftArr,
				cent *centroidDataset)
{
	// index variables
	unsigned int clustIndex, grpIndex;

	double compDistance;
	double minDistance = pointPtr->uprBound;

	for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
	{
		// if the group is not blocked by group filter
		if(groupArr[grpIndex])
		{
			// reset the lwrBoundArr to be only new lwrBounds
			pointPtr->lwrBoundArr[grpIndex] = INFINITY;
		}
		// check to make sure group is not empty and operation won't cause overflow
		else if(pointPtr->lwrBoundArr[grpIndex] != INFINITY)
		{
			pointPtr->lwrBoundArr[grpIndex] -= driftArr[grpIndex];
		}
	}
	
	for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
	{
		// if the centroid's group is marked in groupArr
		if(groupArr[centroidDataset[clustIndex].groupNum])
		{
			// if it was the originally assigned cluster, no need to calc dist
			if(clustIndex == pointPtr->oldCentroid)
			{
				continue;
			}
		
			// compute distance between point and centroid
			compDistance = calcDisCPU(pointPtr->vec, 
									  centroidDataset[clustIndex].vec,
									  NDIM);
			if(compDistance < minDistance)
			{
				pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] =
								 minDistance;
				minDistance = compDistance;
				pointPtr->centroidIndex = clustIndex;
				pointPtr->uprBound = compDistance;
			}
			else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[clustIndex].groupNum])
			{
				pointPtr->lwrBoundArr[centroidDataset[clustIndex].groupNum] = compDistance;
			}
		}

	}

	
}




/*
Function used to do an intial iteration of K-means
*/
void initPoints(point *dataset, 
				cent *centroidDataset)
{

	int pntIndex, clustIndex;
	double currMin;
	double currDistance;
	unsigned int centGroup;
	
	// start single standard k-means iteration for initial bounds and cluster assignments
		// assignment
	for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
	{
		currMin = INFINITY;

		// for all centroids
		for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
		{
			// currDistance is equal to the distance between the current feature
			// vector being inspected, and the current centroid being compared
			currDistance = calcDisCPU(dataset[pntIndex].vec, 
									  centroidDataset[clustIndex].vec,
									  NDIM);

			// if the the currDistance is less than the current minimum distance
			if(currDistance < currMin)
			{
				if(currMin != INFINITY)
				{
					centGroup = 
						centroidDataset[dataset[pntIndex].centroidIndex].groupNum;
					dataset[pntIndex].lwrBoundArr[centGroup] = currMin;					
				}
				// update assignment and upper bound
				dataset[pntIndex].centroidIndex = clustIndex;
				dataset[pntIndex].uprBound = currDistance;

				// update currMin
				currMin = currDistance;
			}
			else
			{
				centGroup = centroidDataset[clustIndex].groupNum;
				if(currDistance < dataset[pntIndex].lwrBoundArr[centGroup])
				{
					dataset[pntIndex].lwrBoundArr[centGroup] = currDistance;
				}
			}
			
		}
	}
}



// need:
// oldCentroid
// dataset of points

void updateCentroids(struct point *dataset, 
					 struct cent *centroidDataset, 
					 double *maxDriftArr)
{
	// holds the number of points assigned to each centroid formerly and currently
	int oldCounts[NCLUST];
	int newCounts[NCLUST];


	// comparison variables
	double compDrift;

	// holds the new vector calculated
	vector oldVec;
	double oldFeature;

	omp_set_num_threads(NTHREAD);

	omp_lock_t driftLock;

	omp_init_lock(&driftLock);
	

	// allocate data for new and old vector sums
	vector *oldSumPtr = 
			(struct vector *)malloc(sizeof(struct vector)*NCLUST);
	vector *newSumPtr = 
			(struct vector *)malloc(sizeof(struct vector)*NCLUST);


	double oldSumFeat;
	double newSumFeat;

	unsigned int pntIndex, clustIndex, grpIndex, dimIndex;
	
	for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
	{
		maxDriftArr[grpIndex] = 0.0;
	}
	for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
	{
		for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
		{
			oldSumPtr[clustIndex].feat[dimIndex] = 0.0;
			newSumPtr[clustIndex].feat[dimIndex] = 0.0;
		}
	
		oldCounts[clustIndex] = 0;
		newCounts[clustIndex] = 0;
	}
	
	for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
	{
		// add one to the old count and new count for each centroid
		if(dataset[pntIndex].oldCentroid >= 0)
		oldCounts[dataset[pntIndex].oldCentroid]++;
		
		newCounts[dataset[pntIndex].centroidIndex]++;

		// if the old centroid does not match the new centroid,
		// add the points vector to each 
		if(dataset[pntIndex].oldCentroid != dataset[pntIndex].centroidIndex)
		{
			for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
			{
				if(dataset[pntIndex].oldCentroid >= 0)
				oldSumPtr[dataset[pntIndex].oldCentroid].feat[dimIndex] += 
									dataset[pntIndex].vec.feat[dimIndex];

				newSumPtr[dataset[pntIndex].centroidIndex].feat[dimIndex] += 
									dataset[pntIndex].vec.feat[dimIndex];
			}
		}
	}

	#pragma omp parallel private(clustIndex,dimIndex,oldVec,oldFeature,oldSumFeat,newSumFeat, compDrift) shared(driftLock,centroidDataset, maxDriftArr)
	{
	// create new centroid points
		#pragma omp for schedule(static)
		for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
		{
			for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
			{
				if(newCounts[clustIndex] > 0)
				{
					oldVec.feat[dimIndex] = centroidDataset[clustIndex].vec.feat[dimIndex];
					oldFeature = oldVec.feat[dimIndex];
					oldSumFeat = oldSumPtr[clustIndex].feat[dimIndex];
					newSumFeat = newSumPtr[clustIndex].feat[dimIndex];

					centroidDataset[clustIndex].vec.feat[dimIndex] = 
					(oldFeature * oldCounts[clustIndex] - oldSumFeat + newSumFeat) 
															/ newCounts[clustIndex];
				}
				else
				{
					// if the centroid has no current members, no change occurs to its position
					oldVec.feat[dimIndex] = centroidDataset[clustIndex].vec.feat[dimIndex];
				}
			}
			

			compDrift = calcDisCPU(oldVec, centroidDataset[clustIndex].vec, NDIM);
			omp_set_lock(&driftLock);
			if(compDrift > maxDriftArr[centroidDataset[clustIndex].groupNum])
			{
				maxDriftArr[centroidDataset[clustIndex].groupNum] = compDrift;
			}
			omp_unset_lock(&driftLock);
			centroidDataset[clustIndex].drift = compDrift;



		}
	}
	omp_destroy_lock(&driftLock);

	free(newSumPtr);
	free(oldSumPtr);
}
