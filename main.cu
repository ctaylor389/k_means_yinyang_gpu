#include "main.h"

int main(int argc, char *argv[])
{
	omp_set_num_threads(1);

	//////////////////////////////////////////////////////////////////
	// Get information from command line
	//1) The dataset filename,
	//2) flag for displaying number of distance calculations performed	
	//3) data dimensionality
	//4) Algorithm Code
	//   0 Yinyang GPU (two filter yinyang)
	//   1 Yinyang CPU (two filter yinyang)
	//   2 Lloyd GPU
	//   3 Lloyd CPU
	//   4 YYSS GPU (single filter yinyang)
	//////////////////////////////////////////////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=4)
	{
		printf("\n\nIncorrect number of input parameters.  \nShould be dataset filename, data dimensionality, algorithm code\n");
		printf("\n 0 for Yinyang GPU 2 filters, 1 for Yinyang CPU, 2 for Lloyd's GPU, 3 for Lloyd's CPU, and 4 for Yinyang GPU 1 filter\n");
		return 1;
	}

	// performance metric variables
	double startTime;
	double endTime;

	//copy parameters from commandline:
	char inputFname[500];
	char inputNumDim[100];
	char inputAlgoCode[100];

	strcpy(inputFname,argv[1]);
	strcpy(inputNumDim,argv[2]);
	strcpy(inputAlgoCode,argv[3]);;

	
	unsigned int numDims = atoi(inputNumDim);
	unsigned int algoCode = atoi(inputAlgoCode);
	unsigned long long int distCalcCount = 0;


	int errFlag;

	unsigned int ranIter;
	

	// check dim to ensure correct operation
	if (numDims!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		passed into the computer program on the command line. 	NDIM=%d, numDims=%d Exiting!!!",NDIM,numDims);
		return 0;
	}


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

		case YINCODE:

			// warm up GPU for time trialing
			warmupGPU();
		
			startYinYangOnGPU(dataset,
							  centroidDataset,
							  &distCalcCount,
							  &startTime,
							  &endTime,
							  &ranIter);

			printf("Yinyang GPU Runtime: %f\n", endTime - startTime);
			
			break;

		case YINCODECPU:
			startYinyangOnCPU(dataset,
							  centroidDataset,
							  &distCalcCount,
							  &startTime,
							  &endTime,
							  &ranIter);

			printf("Yinyang CPU Runtime: %f\n", endTime - startTime);
			
			break;

		case LLOYDCODE:

			// warm up GPU for time trialing
			warmupGPU();
		
			startLloydOnGPU(dataset,
							centroidDataset,
							&startTime,
							&endTime,
							&ranIter);

			printf("Lloyd GPU Runtime: %f\n", endTime - startTime);
						
			break;

		case LLOYDCODECPU:
			startLloydOnCPU(dataset,
							centroidDataset,
							&startTime,
							&endTime,
							&ranIter);

			printf("Lloyd CPU Runtime: %f\n", endTime - startTime);
						
			break;

		case HAMCODE:

			// warm up GPU for time trialing
			warmupGPU();
		
			startHamerlyOnGPU(dataset,
							  centroidDataset,
							  &distCalcCount,
							  &startTime,
							  &endTime,
							  &ranIter);

			printf("Single Filter Yinyang GPU Runtime: %f\n", endTime - startTime);

			
			break;

	}


	free(centroidDataset);
	free(dataset);




	
	return 0;
}



