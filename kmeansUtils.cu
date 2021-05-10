#include "kmeansUtils.h"


DTYPE calcDisCPU(vector vec1,
				  vector vec2,
				  const unsigned int numDim)
{
	unsigned int index;
	DTYPE total = 0.0;

	for(index = 0; index < numDim; index++)
	{
		total += pow((vec2.feat[index] - vec1.feat[index]), 2);
	}


	return sqrt(total);
	
}

int writeTimeData(const char *fname,
				  double *timeArr,
				  int numRuns,
				  int totalIter,
				  int numDim,
				  int numPnt,
				  int numClust,
				  int numGrp,
				  int numThread)
{
	char writeString[300];
	FILE *fp;
	int index;
	double timeSum = 0.0;
	double finalTime;

	// calculate final averaged time
	for(index = 0; index < numRuns; index++)
	{
		timeSum += timeArr[index];
	}
	finalTime = timeSum / numRuns;

	fp = fopen(fname, "a");

	sprintf(writeString, "%f,%d,%d,%d,%d,%d,%d\n", 
							finalTime,numDim,numPnt,numClust,numGrp,numThread,totalIter);
	fputs(writeString, fp);

	
	fclose(fp);
	return 0;
}

int importDataset(const char *fname,
				  point * dataset,
				  const unsigned int numPnt, 
				  const unsigned int numDim)
{

	FILE *fp = fopen(fname, "r");
	if(!fp)
	{
		printf("Unable to open file\n");
		return 1;
	}

	int i;
	int j = 0;
	char fileLine[5000];
	DTYPE feat[numDim];
	char *splitTok;
	

	for(i = 0; i < numPnt; i++)
	{
		if(fgets(fileLine, 5000, fp) == NULL)
		{
			printf("issue on line %d\n", i);
			return 1;
		}
		splitTok = strtok(fileLine, ",");

		while(splitTok != NULL && j < numDim)
		{
			feat[j] = atof(splitTok);
			splitTok = strtok(NULL, ",");
			j++;
		}
		for(j = 0; j < numDim; j++)
		{
			dataset[i].vec.feat[j] = feat[j];
		}
		for(j = 0; j < NGROUPCPU; j++)
		{
			dataset[i].lwrBoundArr[j] = INFINITY;
		}
		j = 0;

		dataset[i].centroidIndex = -1;
		dataset[i].oldCentroid = -1;


	}
	return 0;
}

int generateRandCent(cent *centDataset, 
				 	 const unsigned int numCent,
				 	 const unsigned int numDim,
				 	 const char *filename,
				 	 int seed)
{
	srand(seed + 1);

	unsigned int centIndex, dimIndex;
	DTYPE newFeat;

	char writeString[5000];

	FILE *fp = fopen(filename, "w");
	if(!fp)
	{
		return 1;
	}

	for(centIndex = 0; centIndex < numCent; centIndex++)
	{
		sprintf(writeString, " ");
		for(dimIndex = 0; dimIndex < numDim; dimIndex++)
		{
			newFeat = 2.0*((DTYPE)(rand()) / RAND_MAX);
			centDataset[centIndex].vec.feat[dimIndex] = newFeat;
			if(dimIndex != 0 && dimIndex != numDim)
			{
				sprintf(writeString, "%s,", writeString);
			}
			sprintf(writeString, "%s%f", writeString, newFeat);
		}
		sprintf(writeString, "%s\n", writeString);
		fputs(writeString, fp);
		centDataset[centIndex].groupNum = -1;
		centDataset[centIndex].drift = 0.0;
		centDataset[centIndex].count = 0;
	}
	fclose(fp);
	return 0;
}

int generateCentWithPoint(cent *centDataset,
						  point *dataset,
						  const unsigned int numPnt, 
				 		  const unsigned int numCent,
				 		  const unsigned int numDim)
{
	srand(90);
	int index;
	int dimIndex;
	int randomMax = numPnt / numCent;
	for(index = 0; index < numCent; index++)
	{
		for(dimIndex = 0; dimIndex < numDim; dimIndex++)
		{
			centDataset[index].vec.feat[dimIndex] = 
					dataset[(index * randomMax) + 
							(rand() % randomMax)].vec.feat[dimIndex];
		}
		centDataset[index].groupNum = -1;
		centDataset[index].drift = 0.0;
		centDataset[index].count = 0;
	}
	return 0;
}

int generateCentWithPoint(cent *centDataset,
						  cent *dataset,
   						  const unsigned int numPnt, 
				 		  const unsigned int numCent,
				 		  const unsigned int numDim)
{
	srand(90);
	int index;
	int dimIndex;
	int randomMax = numPnt / numCent;
	for(index = 0; index < numCent; index++)
	{
		for(dimIndex = 0; dimIndex < numDim; dimIndex++)
		{
			centDataset[index].vec.feat[dimIndex] = 
					dataset[(index * randomMax) + 
							(rand() % randomMax)].vec.feat[dimIndex];
		}
		centDataset[index].groupNum = -1;
		centDataset[index].drift = 0.0;
		centDataset[index].count = 0;
	}
	return 0;
}


int groupCent(cent *centDataset, 
			  const unsigned int numClust,
			  const unsigned int numGrps,
			  const unsigned int numDim)
{
	cent *overCent = (cent *)malloc(sizeof(struct cent) * numGrps);
	generateCentWithPoint(overCent, centDataset, numClust, numGrps, numDim);

	unsigned int iterIndex, centIndex, groupIndex, dimIndex, assignment;



	DTYPE currMin = INFINITY;
	DTYPE currDistance = INFINITY;
	DTYPE origVec[numGrps][numDim];

	for(iterIndex = 0; iterIndex < 5; iterIndex++)
	{
		// assignment
		for(centIndex = 0; centIndex < numClust; centIndex++)
		{
			for(groupIndex = 0; groupIndex < numGrps; groupIndex++)
			{
				
				currDistance = calcDisCPU(centDataset[centIndex].vec,
										  overCent[groupIndex].vec,
										  numDim);
				if(currDistance < currMin)
				{
					centDataset[centIndex].groupNum = groupIndex;
					currMin = currDistance;
				}
			}
			currMin = INFINITY;
		}
		// update over centroids
		for(groupIndex = 0; groupIndex < numGrps; groupIndex++)
		{
			for(dimIndex = 0; dimIndex < numDim; dimIndex++)
			{
				origVec[groupIndex][dimIndex] = 
					overCent[groupIndex].vec.feat[dimIndex];
				overCent[groupIndex].vec.feat[dimIndex] = 0.0;
			}
			overCent[groupIndex].count = 0;
		}

		// update over centroids to be average of group
		for(centIndex = 0; centIndex < numClust; centIndex++)
		{
			assignment = centDataset[centIndex].groupNum;
			overCent[assignment].count += 1;
			for(dimIndex = 0; dimIndex < numDim; dimIndex++)
			{
				overCent[assignment].vec.feat[dimIndex] +=
					centDataset[centIndex].vec.feat[dimIndex];
			}



		}


		for(groupIndex = 0; groupIndex < numGrps; groupIndex++)
		{
			if(overCent[groupIndex].count > 0)
			{
				for(dimIndex = 0; dimIndex < numDim; dimIndex++)
				{
					overCent[groupIndex].vec.feat[dimIndex] /=
										overCent[groupIndex].count;
				}				
			}
			else
			{
				for(dimIndex = 0; dimIndex < numDim; dimIndex++)
				{
					overCent[groupIndex].vec.feat[dimIndex] =
									origVec[groupIndex][dimIndex];
				}
			}
		}
	}


	free(overCent);
	return 0;
}



int writeResults(point *dataset, 
				 const unsigned int numPnt,
				 const char *filename)
{
	char writeString[100];
	FILE *fp;
	int i;

	fp = fopen(filename, "w");

	for(i = 0; i < numPnt; i++)
	{
		sprintf(writeString, "%d\n", dataset[i].centroidIndex);
		fputs(writeString, fp);
	}
	fclose(fp);
	return 0;
}

int writeCent(cent *dataset, 
			  const unsigned int numCent,
			  const unsigned int numDim,
			  const char *filename)
{
	char writeString[2000];
	FILE *fp;
	int i, j;

	fp = fopen(filename, "w");

	for(i = 0; i < numCent; i++)
	{
		for(j = 0; j < numDim; j++)
		{
			if(j != 0)
			{
				sprintf(writeString, "%s,%f", writeString, dataset[i].vec.feat[j]);
			}
			else
			{
				sprintf(writeString, "%f", dataset[i].vec.feat[j]);
			}
		}

		sprintf(writeString, "%s\n", writeString);
		fputs(writeString, fp);
	}
	fclose(fp);
	return 0;
}


