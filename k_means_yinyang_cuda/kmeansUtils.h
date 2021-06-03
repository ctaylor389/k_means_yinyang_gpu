#ifndef KMEANS_H
#define KMEANS_H



#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include "params.h"



DTYPE calcDisCPU(vector vec1,
				  vector vec2,
				  const unsigned int numDim);

int writeTimeData(const char *fname,
				  double *timeArr,
				  int numRuns,
				  int totalIter,
				  int numDim,
				  int numPnt,
				  int numClust,
				  int numGrp,
				  int numThread);


int importDataset(const char *fname,
				  point * dataset,
				  const unsigned int numPnt,  
				  const unsigned int numDim);


int generateRandCent(cent *centDataset, 
				 	 const unsigned int numCent,
				 	 const unsigned int numDim,
				 	 const char *filename,
				 	 int seed);

int generateCentWithPoint(cent *centDataset,
						  point *dataset, 
						  const unsigned int numPnt,
				 		  const unsigned int numCent,
				 		  const unsigned int numDim);
	
int generateCentWithPoint(cent *centDataset,
						  cent *dataset, 
						  const unsigned int numPnt,
				 		  const unsigned int numCent,
				 		  const unsigned int numDim);

int groupCent(cent *centDataset, 
			  const unsigned int numClust,
			  const unsigned int numGrps,
			  const unsigned int numDim);


int writeResults(point *dataset, 
				 const unsigned int numPnt,
				 const char *filename);


int writeCent(cent *dataset, 
		   	  const unsigned int numCent,
			  const unsigned int numDim,
			  const char *filename);
	


#endif 
