#ifndef KMEANS_H
#define KMEANS_H



#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include "params.h"



DTYPE calcDisCPU(DTYPE *vec1, DTYPE *vec2, const int numDim);

int writeTimeData(const char *fname, 
                  double *timeArr, 
                  int numRuns, 
                  int totalIter, 
                  int numPnt, 
                  int numCent, 
                  int numGrp,
                  int numDim, 
                  int numThread);


int importPoints(const char *fname,
                 PointInfo *pointInfo,
                 DTYPE *pointData,
                 const int numPnt,
                 const int numDim);

int importData(DTYPE *data, 
               const int numVec,
               const int numDim,
               const char *filename);

int generateRandCent(CentInfo *centInfo,
                     DTYPE *centData,
                     const int numCent,
                     const int numDim,
                     const char *filename,
                     int seed);

int generateCentWithData(CentInfo *centInfo,
                         DTYPE *centData,
                         DTYPE *copyData,
                         const int numCent,
                         const int numCopy,
                         const int numDim);

int groupCent(CentInfo *centInfo,
              DTYPE *centData,
              const int numCent,
              const int numGrp,
              const int numDim);


int writeResults(PointInfo *pointInfo,
                 const int numPnt,
                 const char *filename);


int writeData(DTYPE *data, 
              const int numVec,
              const int numDim,
              const char *filename);
	
ImpType parseImpString(const char *impString);

int compareData(DTYPE *data1,
                DTYPE *data2,
                DTYPE tolerance,
                const int numVec,
                const int numFeat);
                
int compareAssign(PointInfo *info1,
                  PointInfo *info2,
                  const int numPnt);

#endif 
