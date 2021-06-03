#include "main.h"

int main(int argc, char *argv[])
{
    omp_set_num_threads(1);

    ///////////////////////////////////////////////////////////////////////////////
    // Get information from command line
    //1) the dataset filename,
    //2) number of datapoints in dataset
    //3) data dimensionality
    //4) algorithm code
    //   0 Full Yinyang GPU
    //   1 Full Yinyang CPU
    //   2 Simplified Yinyang GPU
    //   3 Simplified Yinyang CPU
    //   4 Super Simplified Yinyang GPU
    //   5 Super Simplified Yinyang CPU //IN PROGRESS//
    //   6 Lloyd GPU
    //   7 Lloyd CPU
    //5) verbose flag 
    //   0 output no additional information
    //   1 output additional information at the cost of performance
    //      NOTE: VERBOSE NOT FOR TIME SENSITIVE EXECUTIONS 
    //6) filename to write time data to for performance measurement
    //   If a file should not be written, use NULL
    ///////////////////////////////////////////////////////////////////////////////

    //Read in parameters from file:
    //dataset filename and cluster instance file
    if (argc!=7)
    {
        printf("\nIncorrect number of input parameters.\n\n"
                "Usage: \n"
                "./kmeans dataset_filename "
                "number_of_lines "
                "data_dimensionality "
                "algorithm_code "
                "verbose_flag "
                "output_time_data_filename\n\n");
        printf("Valid Algorithm Codes:\n"
               "%d for Yinyang Full GPU\n"
               "%d for Yinyang Full CPU\n"
               "%d for Yinyang Simplified GPU\n"
               "%d for Yinyang Simplified CPU\n"
               "%d for Yinyang super Simplified GPU\n"
               "%d for Yinyang super Simplified GPU\n"
               "%d for Lloyd's GPU\n" 
               "%d for Lloyd's CPU\n"
               "\nExiting...\n\n",
               FULLCODE, FULLCODECPU, SIMPLECODE, SIMPLECODECPU,
               SUPERCODE, SUPERCODECPU, LLOYDCODE, LLOYDCODECPU);
        return 1;
    }

    // performance metric variables
    double startTime;
    double endTime;
    unsigned long long int distCalcCount = 0;
    unsigned long long int *countPtr;

    // flags
    int errFlag = 0;
    int writeTimeFlag = 1;
    int verFlag = 0;

    //copy parameters from commandline:
    char inputFname[500];
    char inputVerFlag[100];
    char inputWFname[500];
    char inputNumLines[100];
    char inputNumDim[100];
    char inputAlgoCode[100];

    strcpy(inputFname,argv[1]);
    strcpy(inputNumLines,argv[2]);
    strcpy(inputNumDim,argv[3]);
    strcpy(inputAlgoCode,argv[4]);
    strcpy(inputVerFlag,argv[5]);
    strcpy(inputWFname,argv[6]);
    
    unsigned int numDims = atoi(inputNumDim);
    unsigned int numLines = atoi(inputNumLines);
    int algoCode = atoi(inputAlgoCode);
    verFlag = atoi(inputVerFlag);

    // set flag to write time data
    if(strcmp(inputWFname, "NULL") == 0)
    writeTimeFlag = 0;

    // set distance calc pointer
    if(verFlag)
    countPtr = &distCalcCount;
    else
    countPtr = NULL;

    unsigned int ranIter;

    // array holding the time data to write out
    double timeArr[NRUNS];

    // check dim to ensure correct operation
    if(numDims!=NDIM){
       printf("\nERROR: The number of dimensions defined for the "
              "GPU is not the same as the number of dimensions "
              "passed into the computer program on the "
              "command line.\n"
              "NDIM=%d, numDims=%d Exiting...\n",NDIM,numDims);
        return 1;
    }
    if(numLines!=NPOINT){
       printf("\nERROR: The number of features defined for the "
              "GPU is not the same as the number of features "
              "passed into the computer program on the "
              "command line.\n"
              "NPOINT=%d, numLines=%d Exiting...\n",NPOINT,numLines);
        return 1;
    }

    if(algoCode < FULLCODE || algoCode > LLOYDCODECPU)
    {
        printf("Algorithm code is not valid, "
                "should be an integer between %d and %d.\n"
                "Exiting...\n", FULLCODE, LLOYDCODECPU);
        return 1;
    }
    for(int index = 0; index < NRUNS; index++)
    {

        //import and create dataset
        point *dataset = (point *)malloc(sizeof(point) * NPOINT);

        errFlag = importDataset(inputFname, dataset, NPOINT, NDIM);
        
        if(errFlag)
        {
            // signal erroneous exit
            printf("\nERROR: could not import the dataset, please check file location. Exiting program.\n");		
            return 1;
        }
        
        // allocate centroids
        cent *centroidDataset = (cent *)malloc(sizeof(cent) * NCLUST);

        // generate centroid data using dataset points
        errFlag = generateCentWithPoint(centroidDataset, dataset, NPOINT, NCLUST, NDIM);

        if(errFlag)
        {
            // signal erroneous exit
            printf("\nERROR: Could not generate centroids. Exiting program.\n");		
            return 1;
        }

        switch(algoCode)
        {
            case FULLCODE:
                // warm up GPU for time trialing
                warmupGPU();
                startFullOnGPU(dataset,
                               centroidDataset,
                               countPtr,
                               &startTime,
                               &endTime,
                               &ranIter);

                timeArr[index] = (endTime - startTime) / ranIter;
                
                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                printf("Yinyang Full GPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentFullGPU.txt");

                break;
            case FULLCODECPU:

                if(verFlag)
                {
                    startFullOnCPU(dataset,
                                   centroidDataset,
                                   countPtr,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }
                else
                {
                    startFullOnCPU(dataset,
                                   centroidDataset,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }

                timeArr[index] = (endTime - startTime) / ranIter;
                
                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                
                printf("Yinyang Full CPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentFullCPU.txt");

                break;
            case SIMPLECODE:
                // warm up GPU for time trialing
                warmupGPU();

                startSimpleOnGPU(dataset,
                                 centroidDataset,
                                 countPtr,
                                 &startTime,
                                 &endTime,
                                 &ranIter);

                timeArr[index] = (endTime - startTime) / ranIter;
                
                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                printf("Yinyang Simplified GPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentSimpGPU.txt");
                
                break;
            case SIMPLECODECPU:
                if(verFlag)
                {
                    startSimpleOnCPU(dataset,
                                   centroidDataset,
                                   countPtr,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }
                else
                {
                    startSimpleOnCPU(dataset,
                                   centroidDataset,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }

                
                timeArr[index] = (endTime - startTime) / ranIter;

                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                printf("Yinyang Simplified CPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentSimpCPU.txt");
                
                break;
            case SUPERCODE:
                // warm up GPU for time trialing
                warmupGPU();

                startSuperOnGPU(dataset,
                                centroidDataset,
                                countPtr,
                                &startTime,
                                &endTime,
                                &ranIter);

                timeArr[index] = (endTime - startTime) / ranIter;

                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                printf("Yinyang Super Simplified GPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentSupGPU.txt");
                
                break;         
            case SUPERCODECPU:
                if(verFlag)
                {
                    startSuperOnCPU(dataset,
                                   centroidDataset,
                                   countPtr,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }
                else
                {
                    startSuperOnCPU(dataset,
                                   centroidDataset,
                                   &startTime,
                                   &endTime,
                                   &ranIter);
                }

                timeArr[index] = (endTime - startTime) / ranIter;
                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            distCalcCount);
                }
                printf("Yinyang Super Simplified CPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentSupCPU.txt");
                
                break;
            case LLOYDCODE:

                // warm up GPU for time trialing
                warmupGPU();
            
                startLloydOnGPU(dataset,
                                centroidDataset,
                                &startTime,
                                &endTime,
                                &ranIter);

                
                timeArr[index] = (endTime - startTime) / ranIter;

                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            (unsigned long long int)NCLUST * NPOINT * ranIter);
                }
                printf("Lloyd's GPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentLloydGPU.txt");

                break;
            case LLOYDCODECPU:
                startLloydOnCPU(dataset,
                                centroidDataset,
                                &startTime,
                                &endTime,
                                &ranIter);

                //printf("Lloyd CPU Runtime: %f\n", endTime - startTime);
                timeArr[index] = (endTime - startTime) / ranIter;

                if(verFlag)
                {
                    printf("Total distance calculations performed: %llu\n",
                            (unsigned long long int)NCLUST * NPOINT * ranIter);
                }
                printf("Lloyd's CPU Runtime: %f\n", endTime - startTime);
                writeCent(centroidDataset, NCLUST, NDIM, "FinalCentLloydCPU.txt");

                break;
        }

        free(centroidDataset);
        free(dataset);

    }


    // write timeData
    if(writeTimeFlag)
    {
        writeTimeData(inputWFname, timeArr, NRUNS, ranIter,
                        NDIM, NPOINT, NCLUST, NGROUPCPU, NTHREAD);
    }
    
    return 0;
}
