#include "kmeansTest.h"





int runValidationTests(ImpType impCode)
{
  int failFlag = 0;
  int testCounter = 0;
  
  TestError testResult;
  
  for(int i = 1; i < 5; i++)
  {
    printf("\n");
    
    printf("Starting Test %d with data at ", testCounter);
    testResult = testImpWithKeyImp(SIMPLEGPU, SIMPLEGPU, 1000000, 200, 20, 
                                   32, 500, 16, 1, i, 0.0001, THIRTY_TWO_PATH, 1);
    printErrorMessage(testResult, testCounter);
    testCounter++;
    printf("\n");
    
    if(testResult != testSuccess)
    failFlag = 1;
  }
  
  for(int i = 1; i < 5; i++)
  {
    printf("\n");
    
    printf("Starting Test %d with data at ", testCounter);
    testResult = testImpWithKeyImp(FULLGPU, FULLGPU, 1000000, 200, 20, 
                                   32, 500, 16, 1, i, 0.0001, THIRTY_TWO_PATH, 1);
    printErrorMessage(testResult, testCounter);
    testCounter++;
    printf("\n");
    
    if(testResult != testSuccess)
    failFlag = 1;
  }
  
  for(int i = 1; i < 5; i++)
  {
    printf("\n");
    
    printf("Starting Test %d with data at ", testCounter);
    testResult = testImpWithKeyImp(SUPERGPU, SUPERGPU, 1000000, 200, 20, 
                                   32, 500, 16, 1, i, 0.0001, THIRTY_TWO_PATH, 1);
    printErrorMessage(testResult, testCounter);
    testCounter++;
    printf("\n");
    
    if(testResult != testSuccess)
    failFlag = 1;
  }
  for(int i = 1; i < 5; i++)
  {
    printf("\n");
    
    printf("Starting Test %d with data at ", testCounter);
    testResult = testImpWithKeyImp(LLOYDGPU, LLOYDGPU, 1000000, 200, 20, 
                                   32, 500, 16, 1, i, 0.0001, THIRTY_TWO_PATH, 1);
    printErrorMessage(testResult, testCounter);
    testCounter++;
    printf("\n");
    
    if(testResult != testSuccess)
    failFlag = 1;
  }
  
  return failFlag;
  
}


// handles all build up and teardown of an implementation
// mainly to avoid the switch statements in a hundred different functions
// data structure arguments are double pointers to allow the function to handle mem allocation
  // but also let it pass the info back afterwards (i.e. caller must handle freeing this memory)
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
                          unsigned long long int *countPtr)
{
  
  if(*pointInfo == NULL)
  *pointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);

  if(*pointData == NULL)
  *pointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);

  // import dataset
  if(importPoints(filepath, *pointInfo, *pointData, numPnt, numDim))
  return importError;

  if(*centInfo == NULL)
  *centInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);

  if(*centData == NULL)
  *centData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  if(generateCentWithData(*centInfo, *centData, *pointData, numCent, numPnt, numDim))
  return centGenError;
  
  
  if(countPtr == NULL)
  {
    switch(imp)
    {
      case FULLGPU:
        warmupGPU(numGPU);
        *runtime = startFullOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                       numPnt, numCent, numGrp, numDim, maxIter, numGPU, ranIter);
        break;
      case SIMPLEGPU:
        warmupGPU(numGPU);
        *runtime = startSimpleOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                         numPnt, numCent, numGrp, numDim, maxIter, numGPU, ranIter);
        break;
      case SUPERGPU:
        warmupGPU(numGPU);
        *runtime = startSuperOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, maxIter, numGPU, ranIter);
        break;
      case LLOYDGPU:
        warmupGPU(numGPU);
        *runtime = startLloydOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, maxIter, numGPU, ranIter);
        break;
      case FULLCPU:
        *runtime = startFullOnCPU(*pointInfo, *centInfo, *pointData, *centData, numPnt, 
                       numCent, numGrp, numDim, numThread, maxIter, ranIter);
        break;
      case SIMPLECPU:
        *runtime = startSimpleOnCPU(*pointInfo, *centInfo, *pointData, *centData, numPnt, 
                         numCent, numGrp, numDim, numThread, maxIter, ranIter);
        break;
      case SUPERCPU:
        *runtime = startSuperOnCPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, numThread, maxIter, ranIter);
        break;
      case LLOYDCPU:
        *runtime = startLloydOnCPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, numThread, maxIter, ranIter);
        break;
      default: 
        return unknownImpError;
    }
    
  }
  else
  { 
    switch(imp)
    {
      case FULLGPU:
        warmupGPU(numGPU);
        *runtime = startFullOnGPU(*pointInfo, *centInfo, *pointData, *centData, numPnt,
                       numCent, numGrp, numDim, maxIter, numGPU, ranIter, countPtr);
        break;
      case SIMPLEGPU:
        warmupGPU(numGPU);
        *runtime = startSimpleOnGPU(*pointInfo, *centInfo, *pointData, *centData, numPnt, 
                         numCent, numGrp, numDim, maxIter, numGPU, ranIter, countPtr);
        break;
      case SUPERGPU:
        warmupGPU(numGPU);
        *runtime = startSuperOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, maxIter, numGPU, ranIter, countPtr);
        break;
      case LLOYDGPU:
        warmupGPU(numGPU);
        *runtime = startLloydOnGPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, maxIter, numGPU, ranIter, countPtr);
        break;
      case FULLCPU:
        *runtime = startFullOnCPU(*pointInfo, *centInfo, *pointData, *centData, 
                       numPnt, numCent, numGrp, numDim, numThread, maxIter, ranIter);
        break;
      case SIMPLECPU:
        *runtime = startSimpleOnCPU(*pointInfo, *centInfo, *pointData, *centData,
                         numPnt, numCent, numGrp, numDim, numThread, maxIter, ranIter);
        break;
      case SUPERCPU:
        *runtime = startSuperOnCPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, numThread, maxIter, ranIter);
        break;
      case LLOYDCPU:
        *runtime = startLloydOnCPU(*pointInfo, *centInfo, *pointData, *centData,
                        numPnt, numCent, numDim, numThread, maxIter, ranIter);
        break;
      default: 
        return unknownImpError;
    }
  }
  

  return testSuccess;
}


TestError timeImp(ImpType timedImp, 
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
                  const char *writefile)
{

  // create necessary data structures
  PointInfo *pointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);
  DTYPE *pointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);

  // import dataset
  if(importPoints(filepath, pointInfo, pointData, numPnt, numDim))
  return importError;

  CentInfo *centInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);
  DTYPE *centData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  if(generateCentWithData(centInfo, centData, pointData, numCent, numPnt, numDim))
  return centGenError;

  switch(timedImp)
  {
    case FULLGPU:
      warmupGPU(numGPU);
      *runtime = startFullOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                numCent,numGrp,numDim,maxIter,numGPU,ranIter);
      break;
    case SIMPLEGPU:
      warmupGPU(numGPU);
      *runtime = startSimpleOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                  numCent,numGrp,numDim,maxIter,numGPU,ranIter);
      break;
    case SUPERGPU:
      warmupGPU(numGPU);
      *runtime = startSuperOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,maxIter,numGPU,ranIter);
      break;
    case LLOYDGPU:
      warmupGPU(numGPU);
      *runtime = startLloydOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,maxIter,numGPU,ranIter);
      break;
    case FULLCPU:
      *runtime = startFullOnCPU(pointInfo,centInfo,pointData,centData,numPnt,
                                numCent,numGrp,numDim,numThread,maxIter,ranIter);
      break;
    case SIMPLECPU:
      *runtime = startSimpleOnCPU(pointInfo,centInfo,pointData,centData,numPnt,
                                  numCent,numGrp,numDim,numThread,maxIter,ranIter);
      break;
    case SUPERCPU:
      *runtime = startSuperOnCPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,numThread,maxIter,ranIter);
      break;
    case LLOYDCPU:
      *runtime = startLloydOnCPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,numThread,maxIter,ranIter);
      break;
    default: 
      free(pointInfo);
      free(pointData);
      free(centInfo);
      free(centData);
      return unknownImpError;
  }
  if(writefile != NULL)
  writeData(centData, numCent, numDim, writefile);
  free(pointInfo);
  free(pointData);
  free(centInfo);
  free(centData);
  
  return testSuccess;


}
// returns 1 if the test is failed 
// (i.e. an error occurs or the resulting centroid datasets are not equal)
// returns 0 if the test is succeeded
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
                            const int countFlag)
{
  printf("Filepath: %s\n", filepath);
  double keyRuntime;
  double testRuntime;
  unsigned int keyRanIter;
  unsigned int testRanIter;
  
  unsigned long long int keyCount = 0;
  unsigned long long int testCount = 0;
  
  unsigned long long int *keyCountPtr = NULL;
  unsigned long long int *testCountPtr = NULL;
  
  if(countFlag)
  {
    keyCountPtr = &keyCount;
    testCountPtr = &testCount;
  }
  
  printTestParameters(keyImp,testImp,numPnt,numCent,
                      numGrp,numDim,maxIter, 
                      numThread,keyNumGPU, testNumGPU, tolerance);
  


  DTYPE *pointData = NULL;
  
  PointInfo *keyPointInfo = NULL;
  CentInfo *keyCentInfo = NULL;
  DTYPE *keyCentData = NULL;
  printf("  Starting "); printImpName(keyImp); printf(" on %d GPU's/%d threads\n", keyNumGPU, numThread);
  
  chooseAndRunImp(keyImp, &keyPointInfo, &keyCentInfo, &pointData, &keyCentData,
                  numPnt, numCent, numGrp, numDim, maxIter, numThread, keyNumGPU,
                  &keyRuntime, &keyRanIter, filepath, NULL, keyCountPtr);
  
  printf("  "); printImpName(testImp); printf(" complete\n");

  printf("  Key Implementation runtime: %f\n", keyRuntime);
  printf("  Key iterations ran: %d\n", keyRanIter);
  if(countFlag)
  printf("  Key distance calculations: %llu\n", keyCount);
  
  PointInfo *testPointInfo = NULL;
  CentInfo *testCentInfo = NULL;
  DTYPE *testCentData = NULL;
  printf("  Starting "); printImpName(testImp); printf(" on %d GPU's/%d threads\n", testNumGPU, numThread);
  
  chooseAndRunImp(testImp, &testPointInfo, &testCentInfo, &pointData, &testCentData,
                  numPnt, numCent, numGrp, numDim, maxIter, numThread, testNumGPU,
                  &testRuntime, &testRanIter, filepath, NULL, testCountPtr);
  
  printf("  "); printImpName(testImp); printf(" complete\n");

  printf("  Test Implementation runtime: %f\n", testRuntime);
  printf("  Test iterations ran: %d\n", testRanIter);
  if(countFlag)
  printf("  Test distance calculations: %llu\n", testCount);

  int centResult = compareData(keyCentData, testCentData, tolerance, numCent, numDim);
  
  int assignResult = compareAssign(keyPointInfo, testPointInfo, numPnt);
  
  free(keyPointInfo);
  free(keyCentInfo);
  free(keyCentData);
  free(testPointInfo);
  free(testCentInfo);
  free(testCentData);
  free(pointData);
 
  printf("  Found %d mismatching point assignments\n", assignResult);
 
  if(assignResult)
  return testFailedAssign;
  
  if(centResult)
  return testFailedCent;

  return testSuccess;
}

TestError testImpWithKeyFile(ImpType testImp,
                             const int numPnt,
                             const int numCent,
                             const int numGrp,
                             const int numDim,
                             const int maxIter, 
                             const int numThread,
                             const int numGPU,
                             DTYPE tolerance,
                             char *filepath,
                             const char *keyFilepath)
{
  unsigned int ranIter;
  
  // create necessary data structures
  PointInfo *pointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);
  DTYPE *pointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);

  // import dataset
  if(importPoints(filepath, pointInfo, pointData, numPnt, numDim))
  return importError;

  CentInfo *centInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);
  DTYPE *centData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  if(generateCentWithData(centInfo, centData, pointData, numCent, numPnt, numDim))
  return centGenError;

  switch(testImp)
  {
    case FULLGPU:
      warmupGPU(numGPU);
      startFullOnGPU(pointInfo, centInfo, pointData, centData,
                     numPnt, numCent, numGrp, numDim, maxIter, numGPU, &ranIter);
      break;
    case SIMPLEGPU:
      warmupGPU(numGPU);
      startSimpleOnGPU(pointInfo, centInfo, pointData, centData,
                       numPnt, numCent, numGrp, numDim, maxIter, numGPU, &ranIter);
      break;
    case SUPERGPU:
      warmupGPU(numGPU);
      startSuperOnGPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, maxIter, numGPU, &ranIter);
      break;
    case LLOYDGPU:
      warmupGPU(numGPU);
      startLloydOnGPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, maxIter, numGPU, &ranIter);
      break;
    case FULLCPU:
      startFullOnCPU(pointInfo, centInfo, pointData, centData,
                     numPnt, numCent, numGrp, numDim, numThread, maxIter, &ranIter);
      break;
    case SIMPLECPU:
      startSimpleOnCPU(pointInfo, centInfo, pointData, centData,
                       numPnt, numCent, numGrp, numDim, numThread, maxIter, &ranIter);
      break;
    case SUPERCPU:
      startSuperOnCPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, numThread, maxIter, &ranIter);
      break;
    case LLOYDCPU:
      startLloydOnCPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, numThread, maxIter, &ranIter);
      break;
    default: 
      free(pointInfo);
      free(pointData);
      free(centInfo);
      free(centData);
      return unknownImpError;
  }
  
  
  DTYPE *keyCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);
  importData(keyCentData, numCent, numDim, keyFilepath);
  
  int testResult = compareData(keyCentData, centData, tolerance, numCent, numDim);
  
  free(pointInfo);
  free(pointData);
  free(centInfo);
  free(centData);
  free(keyCentData);
  
  
  if(testResult)
  return testFailedCent;
  else
  return testSuccess;
}




void printErrorMessage(TestError errCode, int testNum)
{
  switch(errCode)
  {
    case testSuccess:
      printf("Test %d successful!", testNum);
      printf("\n");
      break;
    case testFailedCent:
      printf("FAILED: Test %d failed\n", testNum);
      printf("  Final centroids did not match within the given tolerance.");
      printf("\n");
      break;
    case testFailedAssign:
      printf("FAILED: Test %d failed\n", testNum);
      printf("  Final Point Cluster Assignments did not match.");
      printf("\n");
      break;
    case importError:
      printf("Error occurred! ");
      printf("  Dataset could not be imported properly.");
      printf("\n");
      break;
    case centGenError:
      printf("Error occurred! ");
      printf("  Centroids failed to generate from given dataset.");
      printf("\n");
      break;
    case unknownImpError:
      printf("Error occurred! ");
      printf("  An unknown implementation was passed to the test function.");
      printf("\n");
      break;
  }
}

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
                         DTYPE tolerance)
{
  printf("\n  Test Implementation: ");
  switch(testImp)
  {
    case FULLGPU:
      printf("Yinyang Full on %d GPU's", testNumGPU);
      break;
    case SIMPLEGPU:
      printf("Yinyang Simplified on %d GPU's", testNumGPU);
      break;
    case SUPERGPU:
      printf("Yinyang Super Simplified on %d GPU's", testNumGPU);
      break;
    case LLOYDGPU:
      printf("Lloyd's on %d GPU's", testNumGPU);
      break;
    case FULLCPU:
      printf("Yinyang Full CPU with %d threads", numThread);
      break;
    case SIMPLECPU:
      printf("Yinyang Simplified CPU with %d threads", numThread);
      break;
    case SUPERCPU:
      printf("Yinyang Super Simplified CPU with %d threads", numThread);
      break;
    case LLOYDCPU:
      printf("Lloyd's CPU with %d threads", numThread);
      break;
  }
  printf("\n  Key Implementation: ");
  switch(keyImp)
  {
    case FULLGPU:
      printf("Yinyang Full on %d GPU's", keyNumGPU);
      break;
    case SIMPLEGPU:
      printf("Yinyang Simplified on %d GPU's", keyNumGPU);
      break;
    case SUPERGPU:
      printf("Yinyang Super Simplified on %d GPU's", keyNumGPU);
      break;
    case LLOYDGPU:
      printf("Lloyd's on %d GPU's", keyNumGPU);
      break;
    case FULLCPU:
      printf("Yinyang Full CPU with %d threads", numThread);
      break;
    case SIMPLECPU:
      printf("Yinyang Simplified CPU with %d threads", numThread);
      break;
    case SUPERCPU:
      printf("Yinyang Super Simplified CPU with %d threads", numThread);
      break;
    case LLOYDCPU:
      printf("Lloyd's CPU with %d threads", numThread);
      break;
  }
  printf("\n  Parameters:\n");
  printf("    Number of data points: %d\n", numPnt);
  printf("    Number of clusters: %d\n", numCent);
  printf("    Number of groups: %d\n", numGrp);
  printf("    Number of dimensions: %d\n", numDim);
  printf("    Maximum iterations: %d\n", maxIter);
  printf("    Tolerance: %f\n", tolerance);
}


TestError validateDataImport(const char *inputfname, 
                             const char *outputfname, 
                             const int numPnt, 
                             const int numDim)
{
  
  DTYPE *inputPointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);
  PointInfo *inputPointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);
  
  importPoints(inputfname, inputPointInfo, inputPointData, numPnt, numDim);
  writeData(inputPointData, numPnt, numDim, outputfname);
  
  free(inputPointData);
  free(inputPointInfo);
  
  return testSuccess;
  
}

void printImpName(ImpType imp)
{
  switch(imp)
  {
    case FULLGPU:
      printf("Yinyang Full GPU");
      break;
    case SIMPLEGPU:
      printf("Yinyang Simplified GPU");
      break;
    case SUPERGPU:
      printf("Yinyang Super Simplified GPU");
      break;
    case LLOYDGPU:
      printf("Lloyd's GPU");
      break;
    case FULLCPU:
      printf("Yinyang Full CPU");
      break;
    case SIMPLECPU:
      printf("Yinyang Simplified CPU");
      break;
    case SUPERCPU:
      printf("Yinyang Super Simplified CPU");
      break;
    case LLOYDCPU:
      printf("Lloyd's CPU");
      break;
    default:
      printf("Unknown Implementation");
      break;
  }
}


