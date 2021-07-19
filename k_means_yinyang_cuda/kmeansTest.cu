#include "kmeansTest.h"





int runValidationTests(ImpType impCode)
{
  int failFlag = 0;
  
  TestError testResult;
  
  printf("Testing with data at iono\n");
  testResult = testImpWithKeyImp(LLOYDGPU, impCode, 1864620, 200, 20, 
                                 2, 1, 16, 1, 0.0001, "/home/ctaylor/yinyangKmeans/data/synthetic/iono_20min_2Mpts_2D.txt");
  printErrorMessage(testResult);
  
  if(testResult != testSuccess)
  failFlag = 1;
  
  return failFlag;
  
}

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
      warmupGPU();
      *runtime = startFullOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                numCent,numGrp,numDim,maxIter,ranIter);
      break;
    case SIMPLEGPU:
      warmupGPU();
      *runtime = startSimpleOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                  numCent,numGrp,numDim,maxIter,ranIter);
      break;
    case SUPERGPU:
      warmupGPU();
      *runtime = startSuperOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,maxIter,ranIter);
      break;
    case LLOYDGPU:
      warmupGPU();
      *runtime = startLloydOnGPU(pointInfo,centInfo,pointData,centData,numPnt,
                                 numCent,numDim,maxIter,ranIter);
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
                            const int numGPU,
                            DTYPE tolerance,
                            const char *filepath)
{
  printf("%s\n", filepath);
  double keyRuntime;
  double testRuntime;
  unsigned int keyRanIter;
  unsigned int testRanIter;
  
  printTestParameters(keyImp,testImp,numPnt,numCent,
                      numGrp,numDim,maxIter, 
                      numThread,numGPU,tolerance);
  

  // create necessary data structures
  PointInfo *keyPointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);
  DTYPE *pointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);

  // import dataset
  if(importPoints(filepath, keyPointInfo, pointData, numPnt, numDim))
  return importError;

  CentInfo *keyCentInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);
  DTYPE *keyCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  if(generateCentWithData(keyCentInfo, keyCentData, pointData, numCent, numPnt, numDim))
  return centGenError;
  
  switch(keyImp)
  {
    case FULLGPU:
      warmupGPU();
      printf("    *Starting Full GPU\n");
      keyRuntime = startFullOnGPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                                  numPnt, numCent, numGrp, numDim, maxIter, &keyRanIter);
      printf("    *Full GPU complete.\n");
      break;
    case SIMPLEGPU:
      warmupGPU();
      printf("    *Starting Simplified GPU\n");
      keyRuntime = startSimpleOnGPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                                    numPnt, numCent, numGrp, numDim, maxIter, numGPU, &keyRanIter);
      printf("    *Simplified GPU complete.\n");
      break;
    case SUPERGPU:
      warmupGPU();
      printf("    *Starting Super Simplified GPU\n");
      keyRuntime = startSuperOnGPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                                   numPnt, numCent, numDim, maxIter, &keyRanIter);
      printf("    *Super Simplified GPU complete.\n");
      break;
    case LLOYDGPU:
      warmupGPU();
      printf("    *Starting Lloyd GPU\n");
      keyRuntime = startLloydOnGPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                                   numPnt, numCent, numDim, maxIter, &keyRanIter);
      printf("    *Lloyd GPU complete.\n");
      break;
    case FULLCPU:
      printf("    *Starting Full CPU\n");
      keyRuntime = startFullOnCPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                              numPnt, numCent, numGrp, numDim, numThread, maxIter, &keyRanIter);
      printf("    *Full CPU complete.\n");
      break;
    case SIMPLECPU:
      printf("    *Starting Simplified CPU\n");
      keyRuntime = startSimpleOnCPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                              numPnt, numCent, numGrp, numDim, numThread, maxIter, &keyRanIter);
      printf("    *Simplified CPU complete.\n");
      break;
    case SUPERCPU:
      printf("    *Starting Super Simplified CPU\n");
      keyRuntime = startSuperOnCPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                              numPnt, numCent, numDim, numThread, maxIter, &keyRanIter);
      printf("    *Super Simplified CPU complete.\n");
      break;
    case LLOYDCPU:
      printf("    *Starting Lloyd CPU\n");
      keyRuntime = startLloydOnCPU(keyPointInfo, keyCentInfo, pointData, keyCentData,
                              numPnt, numCent, numDim, numThread, maxIter, &keyRanIter);
      printf("    *Lloyd CPU complete.\n");
      break;
    default: 
      free(keyPointInfo);
      free(pointData);
      free(keyCentInfo);
      free(keyCentData);
      return unknownImpError;
  }
  
  printf("    *Key Implementation runtime: %f\n", keyRuntime);

  // create necessary data structures
  PointInfo *testPointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);

  // import dataset
  if(importPoints(filepath, testPointInfo, pointData, numPnt, numDim))
  return importError;

  CentInfo *testCentInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);
  DTYPE *testCentData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  if(generateCentWithData(testCentInfo, testCentData, pointData, numCent, numPnt, numDim))
  return centGenError;

  switch(testImp)
  {
    case FULLGPU:
      warmupGPU();
      printf("    *Starting Full GPU\n");
      testRuntime = startFullOnGPU(testPointInfo, testCentInfo, pointData, testCentData,
                                  numPnt, numCent, numGrp, numDim, maxIter, &testRanIter);
      printf("    *Full GPU complete.\n");
      break;
    case SIMPLEGPU:
      warmupGPU();
      printf("    *Starting Simplified GPU\n");
      testRuntime = startSimpleOnGPU(testPointInfo, testCentInfo, pointData, testCentData,
                                    numPnt, numCent, numGrp, numDim, maxIter, numGPU, &testRanIter);
      printf("    *Simplified GPU complete.\n");
      break;
    case SUPERGPU:
      warmupGPU();
      printf("    *Starting Super Simplified GPU\n");
      testRuntime = startSuperOnGPU(testPointInfo, testCentInfo, pointData, testCentData,
                                   numPnt, numCent, numDim, maxIter, &testRanIter);
      printf("    *Super Simplified GPU complete.\n");
      break;
    case LLOYDGPU:
      warmupGPU();
      printf("    *Starting Lloyd GPU\n");
      testRuntime = startLloydOnGPU(testPointInfo, testCentInfo, pointData, testCentData,
                                   numPnt, numCent, numDim, maxIter, &testRanIter);
      printf("    *Lloyd GPU complete.\n");
      break;
    case FULLCPU:
      printf("    *Starting Full CPU\n");
      testRuntime = startFullOnCPU(testPointInfo, testCentInfo, pointData, testCentData,
                              numPnt, numCent, numGrp, numDim, numThread, maxIter, &testRanIter);
      printf("    *Full CPU complete.\n");
      break;
    case SIMPLECPU:
      printf("    *Starting Simplified CPU\n");
      testRuntime = startSimpleOnCPU(testPointInfo, testCentInfo, pointData, testCentData,
                              numPnt, numCent, numGrp, numDim, numThread, maxIter, &testRanIter);
      printf("    *Simplified CPU complete.\n");
      break;
    case SUPERCPU:
      printf("    *Starting Super Simplified CPU\n");
      testRuntime = startSuperOnCPU(testPointInfo, testCentInfo, pointData, testCentData,
                              numPnt, numCent, numDim, numThread, maxIter, &testRanIter);
      printf("    *Super Simplified CPU complete.\n");
      break;
    case LLOYDCPU:
      printf("    *Starting Lloyd CPU\n");
      testRuntime = startLloydOnCPU(testPointInfo, testCentInfo, pointData, testCentData,
                              numPnt, numCent, numDim, numThread, maxIter, &testRanIter);
      printf("    *Lloyd CPU complete.\n");
      break;
    default: 
        free(keyPointInfo);
        free(pointData);
        free(keyCentInfo);
        free(keyCentData);
        free(testPointInfo);
        free(testCentInfo);
        free(testCentData);
        return unknownImpError;
  }

  printf("    *Test Implementation runtime: %f\n\n", testRuntime);

  int centResult = compareData(keyCentData, testCentData, tolerance, numCent, numDim);
  
  int assignResult = compareAssign(keyPointInfo, testPointInfo, numPnt);
  
  free(keyPointInfo);
  free(pointData);
  free(keyCentInfo);
  free(keyCentData);
  free(testPointInfo);
  free(testCentInfo);
  free(testCentData);
 
  printf("    *Found %d mismatching point assignments\n", assignResult);
 
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
      warmupGPU();
      startFullOnGPU(pointInfo, centInfo, pointData, centData,
                     numPnt, numCent, numGrp, numDim, maxIter, &ranIter);
      break;
    case SIMPLEGPU:
      warmupGPU();
      startSimpleOnGPU(pointInfo, centInfo, pointData, centData,
                       numPnt, numCent, numGrp, numDim, maxIter, &ranIter);
      break;
    case SUPERGPU:
      warmupGPU();
      startSuperOnGPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, maxIter, &ranIter);
      break;
    case LLOYDGPU:
      warmupGPU();
      startLloydOnGPU(pointInfo, centInfo, pointData, centData,
                      numPnt, numCent, numDim, maxIter, &ranIter);
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




void printErrorMessage(TestError errCode)
{
  switch(errCode)
  {
    case testSuccess:
      printf("\nTest Successful!");
      printf("\n");
      break;
    case testFailedCent:
      printf("\nTest Failed: ") ;
      printf("Final centroids did not match within the given tolerance.");
      printf("\n");
      break;
    case testFailedAssign:
      printf("\nTest Failed: ") ;
      printf("Final Point Cluster Assignments did not match.");
      printf("\n");
      break;
    case importError:
      printf("\nError occurred! ");
      printf("Dataset could not be imported properly.");
      printf("\n");
      break;
    case centGenError:
      printf("\nError occurred! ");
      printf("Centroids failed to generate from given dataset.");
      printf("\n");
      break;
    case unknownImpError:
      printf("\nError occurred! ");
      printf("An unknown implementation was passed to the test function.");
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
                         const int numGPU,
                         DTYPE tolerance)
{
  printf("\n\nTest Implementation: ");
  switch(testImp)
  {
    case FULLGPU:
      printf("Yinyang Full on %d GPU's", numGPU);
      break;
    case SIMPLEGPU:
      printf("Yinyang Simplified on %d GPU's", numGPU);
      break;
    case SUPERGPU:
      printf("Yinyang Super Simplified on %d GPU's", numGPU);
      break;
    case LLOYDGPU:
      printf("Lloyd's on %d GPU's", numGPU);
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
  printf("\nKey Implementation: ");
  switch(keyImp)
  {
    case FULLGPU:
      printf("Yinyang Full on %d GPU's", numGPU);
      break;
    case SIMPLEGPU:
      printf("Yinyang Simplified on %d GPU's", numGPU);
      break;
    case SUPERGPU:
      printf("Yinyang Super Simplified on %d GPU's", numGPU);
      break;
    case LLOYDGPU:
      printf("Lloyd's on %d GPU's", numGPU);
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
  printf("\nParameters:\n");
  printf("  Number of data points: %d\n", numPnt);
  printf("  Number of clusters: %d\n", numCent);
  printf("  Number of groups: %d\n", numGrp);
  printf("  Number of dimensions: %d\n", numDim);
  printf("  Maximum iterations: %d\n", maxIter);
  printf("  Tolerance: %f\n", tolerance);
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


