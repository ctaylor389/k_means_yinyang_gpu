#include <math.h>
#include <stdio.h>
#include "params.h"

__global__ void warmup(unsigned int *tmp);


__global__ void checkConverge(struct point * dataset,
							  unsigned int * conFlag);


__global__ void initRunKernel(struct point * dataset,
							  struct cent * centroidDataset,
							  unsigned long long int *calcCount);

__global__ void clearDriftArr(double *devMaxDriftArr);

__global__ void clearCentCalcData(struct vector *newCentSum,
								  struct vector *oldCentSum,
								  unsigned int *newCentCount,
								  unsigned int *oldCentCount);

__global__ void clearCentCalcDataLloyd(struct vector *newCentSum,
								  unsigned int *newCentCount);

__global__ void assignPointsLloyd(struct point * dataset,
								   struct cent * centroidDataset);

__global__ void assignPointsYinyang(struct point * dataset,
								   struct cent * centroidDataset,
								   double * maxDriftArr,
								   unsigned long long int *calcCount);

__global__ void assignPointsHamerly(struct point * dataset,
								   struct cent * centroidDataset,
								   double * maxDrift,
								   unsigned long long int *calcCount);

__global__ void updateCentroidsKernel(struct point * dataset,
									  struct cent * centroidDataset,
									  double *maxDriftArr);

__global__ void updateCentroidsLloyd(struct point * dataset,
									  struct cent * centroidDataset,
									  double *maxDriftArr);

__global__ void calcCentData(struct point *dataset,
							 struct cent *centroidDataset,
							 struct vector *oldSums,
							 struct vector *newSums,
							 unsigned int *oldCounts,
							 unsigned int *newCounts);

__global__ void calcNewCentroids(struct point *dataset,
								 struct cent *centroidDataset,
								 double *maxDriftArr,
								 struct vector *oldSums,
								 struct vector *newSums,
								 unsigned int *oldCounts,
								 unsigned int *newCounts);

__global__ void calcCentDataLloyd(struct point *dataset,
							 struct cent *centroidDataset,
							 struct vector *newSums,
							 unsigned int *newCounts);

__global__ void calcNewCentroidsLloyd(struct point *dataset,
								 struct cent *centroidDataset,
								 struct vector *newSums,
								 unsigned int *newCounts);

__global__ void calcNewCentroidsAve(struct point *dataset,
								 struct cent *centroidDataset,
								 struct vector *newSums,
								 unsigned int *newCounts,
								 double *maxDriftArr);

__device__ void pointCalcs(struct point *pointPtr,
						   struct cent *centroidDataset,
						   unsigned int *groupArr,
						   double *maxDriftArr,
						   unsigned long long int *calcCount);

__device__ double calcDis(struct vector *vec1, 
						  struct vector *vec2);

__device__ void AtomicMax(const double *address, 
						  const double value);

