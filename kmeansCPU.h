#ifndef KMEANSCPU_H
#define KMEANSCPU_H

#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "params.h"
#include "kmeansUtils.h"

int startFullOnCPU(point *dataset,
				   cent *centroidDataset,
				   double *startTime,
				   double *endTime,
				   unsigned int *ranIter);

int startSimpleOnCPU(point *dataset,
					 cent *centroidDataset,
				  	 double *startTime,  
				  	 double *endTime,
				  	 unsigned int *ranIter);

int startSuperOnCPU(point *dataset,
					cent *centroidDataset,
				  	double *startTime,
				  	double *endTime,
				  	unsigned int *ranIter);

int startLloydOnCPU(point *dataset,
					cent *centroidDataset,
				  	double *startTime,  
				  	double *endTime,
				  	unsigned int *ranIter);

unsigned int checkConverge(point *dataset);

void updateCentroids(point *dataset,
					 cent *centroidDataset,
					 DTYPE *maxDriftArr);

void pointCalcsSimple(point *pointPtr, 
				      int *groupArr,
                      DTYPE *driftArr, 
                      cent *centroidDataset);
				
void pointCalcsFull(point *pointPtr, 
				    int *groupArr,
				    DTYPE *driftArr, 
				    cent *centroidDataset);

void pointCalcsFullAlt(point *pointPtr, 
				       int *groupArr,
				       DTYPE *driftArr, 
				       cent *centroidDataset);

void initPoints(point *dataset, 
				cent *centroidDataset);

// overloads for counting distance calculations

int startFullOnCPU(point *dataset,
				   cent *centroidDataset,
				   unsigned long long int *distCalcCount,
				   double *startTime,
				   double *endTime,
				   unsigned int *ranIter);

int startSimpleOnCPU(point *dataset,
					 cent *centroidDataset,
					 unsigned long long int *distCalcCount,
				  	 double *startTime,  
				  	 double *endTime,
				  	 unsigned int *ranIter);

int startSuperOnCPU(point *dataset,
					cent *centroidDataset,
					unsigned long long int *distCalcCount,
				  	double *startTime,
				  	double *endTime,
				  	unsigned int *ranIter);

unsigned long long int pointCalcsSimpleCount(point *pointPtr, 
				                             int *groupArr,
				                             DTYPE *driftArr, 
				                             cent *centroidDataset);
				
unsigned long long int pointCalcsFullCount(point *pointPtr, 
				                           int *groupArr,
				                           DTYPE *driftArr, 
				                           cent *centroidDataset);

unsigned long long int pointCalcsFullAltCount(point *pointPtr,
				                              int *groupArr,
				                              DTYPE *driftArr,
				                              cent *centroidDataset);

void initPoints(point *dataset, 
				cent *centroidDataset,
				unsigned long long int *distCalcCount);

#endif
