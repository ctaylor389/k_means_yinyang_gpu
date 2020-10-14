#ifndef KMEANSCPU_H
#define KMEANSCPU_H


//#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "params.h"
#include "kmeansUtils.h"

int startLloydOnCPU(point *dataset,
					cent *centroidDataset,
				  	double *startTime,  
				  	double *endTime,
				  	unsigned int *ranIter);

int startYinyangOnCPU(point *dataset,
					  cent *centroidDataset,
					  unsigned long long int *distCalcCount,
				  	  double *startTime,  
				  	  double *endTime,
				  	  unsigned int *ranIter);

unsigned int checkConverge(point *dataset);

void updateCentroids(point *dataset,
					 cent *centroidDataset,
					 double *maxDriftArr);

void pointCalcs(point *pointPtr, 
				int *groupArr,
				double *driftArr, 
				cent *centroidDataset);

void initPoints(point *dataset, 
				cent *centroidDataset);


#endif
