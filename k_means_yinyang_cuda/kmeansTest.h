#ifndef TEST_H
#define TEST_H



#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include "omp.h"
#include "params.h"
#include "kmeansCPU.h"
#include "kmeansUtil.h"
#include "kmeansGPU.h"


typedef enum{
  testSuccess = 0,
  testFailedCent,
  testFailedAssign,
  importError,
  centGenError,
  unknownImpError
  
} TestError;

int runValidationTests(ImpType impCode);

TestError chooseAndRunImp(ImpType imp,
                          PointInfo **pointInfo,
                          CentInfo **centInfo,
                          DTYPE **pointData,
                          DTYPE **centData,
                          const int numPnt,
                          const int numCent,
                          const int numGrp,
                          const int numDim,
                          const int maxIter,
                          const int numThread,
                          const int numGPU,
                          double *runtime,
                          unsigned int *ranIter,
                          const char *filepath,
                          const char *writefile,
                          unsigned long long int *countPtr);

TestError timeImp(ImpType timedImp, 
                  const int numPnt,
                  const int numCent, 
                  const int numGrp,
                  const int numDim,
                  const int maxIter, 
                  const int numThread,
                  double *runtime,
                  unsigned int *ranIter, 
                  const char *filepath,
                  const char *writefile);

TestError testImpWithKeyImp(ImpType keyImp,
                            ImpType testImp,
                            const int numPnt,
                            const int numCent, 
                            const int numGrp,
                            const int numDim,
                            const int maxIter, 
                            const int numThread,
                            const int keyNumGPU,
                            const int testNumGPU,
                            DTYPE tolerance,
                            const char *filepath,
                            const int countFlag);

TestError testImpWithKeyFile(ImpType testImp,
                             const int numPnt,
                             const int numCent, 
                             const int numGrp,
                             const int numDim,
                             const int maxIter, 
                             const int numthread,
                             DTYPE tolerance,
                             const char *filepath, 
                             const char *keyFilepath);



void printErrorMessage(TestError errCode, int testNum);

void printTestParameters(ImpType keyImp,
                         ImpType testImp,
                         const int numPnt,
                         const int numCent, 
                         const int numGrp,
                         const int numDim,
                         const int maxIter, 
                         const int numThread,
                         const int keyNumGPU,
                         const int testNumGPU,
                         DTYPE tolerance);

TestError validateDataImport(const char *inputfname,
                             const char *outputfname, 
                             const int numPnt,
                             const int numDim);
                             

void printImpName(ImpType imp);


#define MSD_PATH "/data/real/YearPredictionMSD.txt"
#define MSD_SIZE 515345
#define MSD_DIM 90



#define SUSY_PATH "/data/real/SUSY_normalize_0_1.txt"
#define SUSY_SIZE 5000000
#define SUSY_DIM 18

#define CENS_PATH "/data/real/USCensus/USCensus1990.data_no_first_col.txt"
#define CENS_SIZE 2458285
#define CENS_DIM 68

#define HIGG_PATH "/data/higgs/higgs_normalize_0_1.txt"
#define HIGG_SIZE 11000000
#define HIGG_DIM 28

#define THIRTY_TWO_PATH "/home/ctaylor/data/synthetic/dataset_fixed_len_pts_expo_NDIM_32_pts_1000000.txt"

#endif