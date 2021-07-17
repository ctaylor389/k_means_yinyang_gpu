#include "kmeansMain.h"

int main(int argc, char *argv[])
{
  omp_set_num_threads(1);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Required Positional Arguements:
  // <algorithm>
  //   Valid Values: FULLGPU, SIMPLEGPU, SUPERGPU, LLOYDGPU, FULLCPU, SIMPLECPU, SUPERCPU, LLOYDCPU
  // <filepath_to_dataset>
  // <number_of_datapoints>
  // <data_dimensionality>
  // <number_of_clusters>
  //
  // Optional Flags:
  // -t  <number_of_groups>      (Must be <= number of clusters. Default = 20 for GPU, k/10 for CPU)
  // -c                          (count distance calculations)
  // -m  <number_of_CPU_threads> (Default is 1)
  // -i  <max_iterations_to_run> (Default is 1000)
  // -g  <number_of_GPUs>        (Default is 1)
  // -wc <write_to_filepath>     (write final clusters to filepath) 
  // -wa <write_to_filepath>     (write final point assignments to filepath)
  // -wt <write_to_filepath>     (write timing data to filepath)
  // -v                          (run validation tests)
  // -h                          (print help info)
  //////////////////////////////////////////////////////////////////////////////////////////////////


  // check if help flag was called
  for(int i = 0; i < argc; i++)
  {
    if(!strcmp(argv[i],"-h"))
    {
      printf("\n////////////////////////////////////////////////////////////////////////////////////////////////\n"
             "// Required Positional Arguements:\n"
             "// <algorithm>\n"
             "//   Valid Values: FULLGPU, SIMPLEGPU, SUPERGPU, LLOYDGPU, FULLCPU, SIMPLECPU, SUPERCPU, LLOYDCPU\n"
             "// <filepath_to_dataset>\n"
             "// <number_of_datapoints>\n"
             "// <data_dimensionality>\n"
             "// <number_of_clusters>\n"
             "\n"
             "// Optional Flags:\n"
             "// -t  <number_of_groups>      (Must be <= number of clusters. Default = 20 for GPU, k/10 for CPU)\n"
             "// -m  <number_of_CPU_threads> (Default is 1)\n"
             "// -i  <max_iterations_to_run> (Default is 1000)\n"
             "// -g  <number_of_GPUs>        (Default is 1)\n"
             "// -wc <write_to_filepath>     (write final clusters to filepath)\n"
             "// -wa <write_to_filepath>     (write final point assignments to filepath)\n"
             "// -wt <write_to_filepath>     (write timing data to filepath)\n"
             "// -v                          (run validation tests)"
             "// -h                          (print help info)\n"
             "//////////////////////////////////////////////////////////////////////////////////////////////////\n\n");
      return 0;
    }
    else if(!strcmp(argv[i],"-v"))
    {
      if(i + 1 >= argc)
      {
        printf("Error: Implementation to test must be provided after -v flag.");
        printf("Exiting...\n");
        return 1;
      }
      ImpType impCode = parseImpString(argv[i + 1]);
      if(impCode == INVALIDIMP)
      {
        printf("Error: Invalid implementation given. Use -h flag for valid implementation types. Exiting...\n");
        printf("Exiting...\n");
        return 1;
      }
      if(runValidationTests(impCode))
      {
        printf("Not all tests succeeded... See above output for which tests succeeded and which failed.\n\n");
      }
      else
      {
        printf("All tests succeeded!\n\n");
      }
      return 0;
    }
  }
  
  int numPnt;
  int numCent;
  int numGrp = 20;
  int numDim;
  int numThread = 1;
  int maxIter = 1000;
  int numGPU = 1;
  double runtime;
  int writeCentFlag = 0;
  int writeAssignFlag = 0;
  int writeTimeFlag = 0;
  char *writeCentPath;
  char *writeAssignPath;
  char *writeTimePath;
  

  //Read in parameters from file:
  //dataset filename and cluster instance file
  if (argc < 5)
  {
    printf("Required arguements were not provided.\n");
    printf("Use the -h flag for valid arguements. Exiting...\n");
    return 1;
  }

  // get the implementation
  ImpType impCode = parseImpString(argv[1]);
  if(impCode == INVALIDIMP)
  {
    printf("Error: Invalid implementation given. Use -h flag for valid implementation types. Exiting...\n"); 
    return 1;
  }
  
  
  
  char datasetPath[200];
  strcpy(datasetPath, argv[2]);

  // get data info
  numPnt = atoi(argv[3]);
  if(numPnt <= 0){
    printf("Error: Cannot recognize given number of points. Exiting...\n"); 
    return 1;
  }
  
  numDim = atoi(argv[4]);
  if(numDim <= 0){
    printf("Error: Cannot recognize given number of dimensions. Exiting...\n"); 
    return 1;
  }
  
  numCent = atoi(argv[5]);
  if(numCent <= 0){
    printf("Error: Cannot recognize given number of clusters. Exiting...\n"); 
    return 1;
  }
  if(numGrp > numCent)
  numGrp = numCent;
  
  if(impCode == FULLCPU || impCode == SIMPLECPU || impCode == SUPERCPU || impCode == LLOYDCPU){
    if(numCent < 10)
    numGrp = numCent;
    else
    numGrp = numCent / 10;
  }
  
  
  // get optional arguments
  for(int i = 0; i < argc; i++)
  {
    if(!strcmp(argv[i],"-t") && i+1 < argc)
    {
      numGrp = atoi(argv[i+1]);
      if(numGrp > numCent)
      {
        printf("Error: Given number of groups invalid. Exiting...\n");
        return 1;
      }
    }
    else if(!strcmp(argv[i],"-m") && i+1 < argc)
    {
      numThread = atoi(argv[i+1]);
      if(numThread <= 0)
      {
        printf("Error: Given number of groups invalid. Exiting...\n");
        return 1;
      }
    }
    else if(!strcmp(argv[i],"-i") && i+1 < argc)
    {
      maxIter = atoi(argv[i+1]);
      if(maxIter <= 0)
      {
        printf("Error: Given max number of iterations invalid. Exiting...\n");
        return 1;
      }
    }
    else if(!strcmp(argv[i],"-g") && i+1 < argc)
    {
      int availGPU;
      hipGetDeviceCount(&availGPU);
      numGPU = atoi(argv[i+1]);
      if(numGPU <= 0 || numGPU > availGPU)
      {
        printf("Error: Invalid number of requested GPU's. Exiting...\n"); 
        return 1;
      }
    }
    else if(!strcmp(argv[i],"-wc") && i+1 < argc)
    {
      writeCentFlag = 1;
      writeCentPath = argv[i+1];
    }
    else if(!strcmp(argv[i],"-wa") && i+1 < argc)
    {
      writeAssignFlag = 1;
      writeAssignPath = argv[i+1];
    }
    else if(!strcmp(argv[i],"-wt") && i+1 < argc)
    {
      writeTimeFlag = 1;
      writeTimePath = argv[i+1];
    }
  }
  
  unsigned int ranIter;


  //import and create dataset
  PointInfo *pointInfo = (PointInfo *)malloc(sizeof(PointInfo) * numPnt);
  DTYPE *pointData = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numDim);
  

  if(importPoints(datasetPath, pointInfo, pointData, numPnt, numDim))
  {
    // signal erroneous exit
    printf("\nERROR: could not import the dataset, please check file location. Exiting program.\n");
    free(pointInfo);
    free(pointData);
    return 1;
  }

  // allocate centroids
  CentInfo *centInfo = (CentInfo *)malloc(sizeof(CentInfo) * numCent);
  DTYPE *centData = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);

  // generate centroid data using dataset points
  if(generateCentWithData(centInfo, centData, pointData, numCent, numPnt, numDim))
  {
    // signal erroneous exit
    printf("\nERROR: Could not generate centroids. Exiting program.\n");
    free(pointInfo);
    free(pointData);
    free(centInfo);
    free(centData);
    return 1;
  }

  switch(impCode)
  {
    case FULLGPU:
      warmupGPU();
      runtime = 
        startFullOnGPU(pointInfo, centInfo, pointData, centData,
                       numPnt, numCent, numGrp, numDim, maxIter, &ranIter);
      break;
    case SIMPLEGPU:
      warmupGPU();
      runtime = 
        startSimpleOnGPU(pointInfo, centInfo, pointData, centData,
                         numPnt, numCent, numGrp, numDim, maxIter, &ranIter);
      break;
    case SUPERGPU:
      warmupGPU();
      runtime = 
        startSuperOnGPU(pointInfo, centInfo, pointData, centData,
                        numPnt, numCent, numDim, maxIter, &ranIter);
      break;
    case LLOYDGPU:
      warmupGPU();
      runtime = 
        startLloydOnGPU(pointInfo, centInfo, pointData, centData,
                        numPnt, numCent, numDim, maxIter, &ranIter);
      break;
    case FULLCPU:
      runtime = 
        startFullOnCPU(pointInfo, centInfo, pointData, centData, numPnt, 
                       numCent, numGrp, numDim, numThread, maxIter, &ranIter);
      break;
    case SIMPLECPU:
      runtime = 
        startSimpleOnCPU(pointInfo, centInfo, pointData, centData, numPnt, 
                         numCent, numGrp, numDim, numThread, maxIter, &ranIter);
      break;
    case SUPERCPU:
      runtime = 
        startSuperOnCPU(pointInfo, centInfo, pointData, centData,
                        numPnt, numCent, numDim, numThread, maxIter, &ranIter);
      break;
    case LLOYDCPU:
      runtime = 
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
  
  if(writeCentFlag)
  writeData(centData, numCent, numDim, writeCentPath);
  if(writeAssignFlag)
  writeResults(pointInfo, numPnt, writeAssignPath);
  if(writeTimeFlag)
  writeTimeData(writeTimePath, &runtime, 1, ranIter, 
                numPnt, numCent, numGrp, numDim, numThread);

  free(pointData);
  free(centData);
  free(pointInfo);
  free(centInfo);

  return 0;
}
