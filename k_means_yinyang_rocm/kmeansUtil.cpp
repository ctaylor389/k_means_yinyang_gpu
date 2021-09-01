#include "kmeansUtil.h"


DTYPE calcDisCPU(DTYPE *vec1, 
                 DTYPE *vec2, 
                 const int numDim)
{
	unsigned int index;
	DTYPE total = 0.0;
  DTYPE square;

	for(index = 0; index < numDim; index++)
	{
    square = (vec1[index] - vec2[index]);
		total += square * square;
	}

	return sqrt(total);
	
}



int writeTimeData(const char *fname,
									const char *impStr,
                  double *timeArr,
                  int numRuns, 
                  int totalIter,
                  int numPnt,
                  int numCent, 
                  int numGrp,
                  int numDim,
                  int numThread,
									int numGPU,
									unsigned long long int calcCount)
{
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
    
  if(!fp)
  return 1;

	fseek(fp, 0, SEEK_END);

	if (ftell(fp) == 0) 
	{
		fprintf(fp, "Implementation, Time, Iterations, Points, Centroids, Groups, Dimensions, CPUThreads, GPUs, CountFlag\n");
	}

	if (calcCount)
	{
		fprintf(fp, "%s, %f, %d, %d, %d, %d, %d, %d, %d, %llu\n", 
          impStr, finalTime, totalIter, numPnt, numCent, numGrp, numDim, numThread, numGPU, calcCount);
	}

	else
	{
		fprintf(fp, "%s, %f, %d, %d, %d, %d, %d, %d, %d, %s\n", 
          impStr, finalTime, totalIter, numPnt, numCent, numGrp, numDim, numThread, numGPU, "N/A");
	}
	
	fclose(fp);
	return 0;
}

int importPoints(const char *fname,
                 PointInfo *pointInfo,
                 DTYPE *pointData,
                 const int numPnt,
                 const int numDim)
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
			pointData[(i * numDim) + j] = feat[j];
		}
		j = 0;

		pointInfo[i].centroidIndex = -1;
		pointInfo[i].oldCentroid = -1;
    pointInfo[i].uprBound = INFINITY;
	}
	return 0;
}


int importData(DTYPE *data, 
               const int numVec,
               const int numDim,
               const char *filename)
{
  FILE *fp = fopen(filename, "r");
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

	for(i = 0; i < numVec; i++)
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
			data[(i * numDim) + j] = feat[j];
		}
		j = 0;
	}
    return 0;
}


int generateRandCent(CentInfo *centInfo,
                     DTYPE *centData,
                     const int numCent,
                     const int numDim,
                     const char *filename,
                     int seed)
{
	srand(seed + 1);

	unsigned int centIndex, dimIndex;
	DTYPE newFeat;


	FILE *fp = fopen(filename, "w");
	if(!fp)
	{
		return 1;
	}

	for(centIndex = 0; centIndex < numCent; centIndex++)
	{
		for(dimIndex = 0; dimIndex < numDim; dimIndex++)
		{
			newFeat = 2.0*((DTYPE)(rand()) / RAND_MAX);
			centData[(centIndex * numDim) + dimIndex] = newFeat;
			
			fprintf(fp, "%f", newFeat);
      if(dimIndex != numDim - 1)
			fprintf(fp, ",");
		}
		fprintf(fp, "\n");
		centInfo[centIndex].groupNum = -1;
		centInfo[centIndex].drift = 0.0;
		centInfo[centIndex].count = 0;
	}
	fclose(fp);
	return 0;
}

int generateCentWithData(CentInfo *centInfo,
                         DTYPE *centData,
                         DTYPE *copyData,
                         const int numCent,
                         const int numCopy,
                         const int numDim)
{
	srand(90);
	int i;
	int j;
	int randomMax = numCopy / numCent;
	for(i = 0; i < numCent; i++)
	{
		for(j = 0; j < numDim; j++)
		{
			centData[(i * numDim) + j] = 
        copyData[((i * randomMax) + 
				(rand() % randomMax)) * numDim + j];
		}
		centInfo[i].groupNum = -1;
		centInfo[i].drift = 0.0;
		centInfo[i].count = 0;
	}
	return 0;
}


int groupCent(CentInfo *centInfo,
              DTYPE *centData,
              const int numCent,
              const int numGrp,
              const int numDim)
{
	CentInfo *overInfo = (CentInfo *)malloc(sizeof(CentInfo) * numGrp);
  DTYPE *overData = (DTYPE *)malloc(sizeof(DTYPE) * numGrp * numDim);
	generateCentWithData(overInfo, overData, centData, numGrp, numCent, numDim);

	unsigned int iterIndex, centIndex, grpIndex, dimIndex, assignment;

	DTYPE currMin = INFINITY;
	DTYPE currDistance = INFINITY;
	DTYPE origVec[numGrp][numDim];

	for(iterIndex = 0; iterIndex < 5; iterIndex++)
	{
		// assignment
		for(centIndex = 0; centIndex < numCent; centIndex++)
		{
			for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
			{
				currDistance = calcDisCPU(&centData[centIndex * numDim],
										  &overData[grpIndex * numDim],
										  numDim);
				if(currDistance < currMin)
				{
					centInfo[centIndex].groupNum = grpIndex;
					currMin = currDistance;
				}
			}
			currMin = INFINITY;
		}
		// update over centroids
		for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
		{
			for(dimIndex = 0; dimIndex < numDim; dimIndex++)
			{
				origVec[grpIndex][dimIndex] = 
					overData[(grpIndex * numDim) + dimIndex];
				overData[(grpIndex * numDim) + dimIndex]= 0.0;
			}
			overInfo[grpIndex].count = 0;
		}

		// update over centroids to be average of group
		for(centIndex = 0; centIndex < numCent; centIndex++)
		{
			assignment = centInfo[centIndex].groupNum;
			overInfo[assignment].count += 1;
			for(dimIndex = 0; dimIndex < numDim; dimIndex++)
			{
				overData[(assignment * numDim) + dimIndex] +=
          centData[(centIndex * numDim) + dimIndex];
			}
		}


		for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
		{
			if(overInfo[grpIndex].count > 0)
			{
				for(dimIndex = 0; dimIndex < numDim; dimIndex++)
				{
					overData[(grpIndex * numDim) + dimIndex] /=
            overInfo[grpIndex].count;
				}				
			}
			else
			{
				for(dimIndex = 0; dimIndex < numDim; dimIndex++)
				{
					overData[(grpIndex * numDim) + dimIndex] =
            origVec[grpIndex][dimIndex];
				}
			}
		}
	}
	free(overData);
  free(overInfo);
	return 0;
}



int writeResults(PointInfo *pointInfo,
				 const int numPnt,
				 const char *filename)
{
	FILE *fp;
	int i;

	fp = fopen(filename, "w");

	for(i = 0; i < numPnt; i++)
	{
		fprintf(fp, "%d\n", pointInfo[i].centroidIndex);
	}
	fclose(fp);
	return 0;
}

int writeData(DTYPE *data, 
			  const int numVec,
			  const int numDim,
			  const char *filename)
{
	FILE *fp;
	int i, j;

	fp = fopen(filename, "w");

	for(i = 0; i < numVec; i++)
	{
		for(j = 0; j < numDim; j++)
		{
			fprintf(fp, "%f", data[(i * numDim) + j]);
      if(j != numDim - 1)
      fprintf(fp, ",");
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	return 0;
}


ImpType parseImpString(const char *impString)
{
  if(!strcmp(impString, "FULLGPU"))
  return FULLGPU;
  else if(!strcmp(impString, "FULLCPU"))
  return FULLCPU;
  else if(!strcmp(impString, "SIMPLEGPU"))
  return SIMPLEGPU;
  else if(!strcmp(impString, "SIMPLECPU"))
  return SIMPLECPU;
  else if(!strcmp(impString, "SUPERGPU"))
  return SUPERGPU;
  else if(!strcmp(impString, "SUPERCPU"))
  return SUPERCPU;
  else if(!strcmp(impString, "LLOYDGPU"))
  return LLOYDGPU;
  else if(!strcmp(impString, "LLOYDCPU"))
  return LLOYDCPU;
  else if(!strcmp(impString, "MULTIGPUTEST"))
  return MULTIGPUTEST;
  else if(!strcmp(impString, "KEYFILE"))
  return KEYFILE;
  else
  return INVALIDIMP;
}



// returns 0 if datasets are equal within the tolerance given
// returns 1 if not
int compareData(DTYPE *data1,
                DTYPE *data2,
                DTYPE tolerance,
                const int numVec,
                const int numFeat)
{
  int index, dimIndex;
  DTYPE feat1, feat2, diff;
  DTYPE maxDiff = 0.0;
  int returnFlag = 0;

  for(index = 0; index < numVec; index++)
  {
    for(dimIndex = 0; dimIndex < numFeat; dimIndex++)
    {
      feat1 = data1[(index * numFeat) + dimIndex];
      feat2 = data2[(index * numFeat) + dimIndex];
      diff = abs(feat1 - feat2);
      if(diff > tolerance)
      returnFlag = 1;
  
      if(diff > maxDiff)
      maxDiff = diff;
    }
  }
  printf("  Max difference between data is %.15f\n", maxDiff);
  
  // datasets are equal within tolerance
  return returnFlag;
}

int compareAssign(PointInfo *info1,
                  PointInfo *info2,
                  const int numPnt)
{
  int mismatch = 0;
  for(int i = 0; i < numPnt; i++)
  {
    if(info1[i].centroidIndex != info2[i].centroidIndex)
    mismatch++;
  }
  return mismatch;
}

void calcWeightedMeans(CentInfo *newCentInfo,
                       CentInfo **allCentInfo,
                       DTYPE *newCentData,
                       DTYPE *oldCentData,
                       DTYPE **allCentData,
                       DTYPE *newMaxDriftArr,
                       const int numCent,
                       const int numGrp,
                       const int numDim,
                       const int numGPU)
{
  DTYPE numerator = 0;
  DTYPE denominator = 0;
  DTYPE zeroNumerator = 0;
  int zeroCount = 0;

  for (int i = 0; i < numCent; i++)
  {
      for (int j = 0; j < numDim; j++)
      {
          oldCentData[(i * numDim) + j] = newCentData[(i * numDim) + j];
      }
  }

  for (int i = 0; i < numGPU; i++)
  {
      for (int j = 0; j < numCent; j++)
      {
        newCentInfo[j].count += allCentInfo[i][j].count;

        newCentInfo[j].groupNum = allCentInfo[0][j].groupNum;
      }
  }

  for (int j = 0; j < numCent; j++)
  {
      for (int k = 0; k < numDim; k++)
      {
          for (int l = 0; l < numGPU; l++)
          {
              if (allCentInfo[l][j].count == 0)
              {
                  zeroCount++;
                  zeroNumerator += allCentData[l][(j * numDim) + k];
              }

              numerator +=
              allCentData[l][(j * numDim) + k]*allCentInfo[l][j].count;

              denominator += allCentInfo[l][j].count;
          }

          if (denominator != 0)
          {
              newCentData[(j * numDim) + k] = numerator/denominator;
          }

          else
          {
              newCentData[(j * numDim) + k] = zeroNumerator/zeroCount;
          }

          zeroCount = 0;
          zeroNumerator = 0;
          numerator = 0;
          denominator = 0;
      }

      newCentInfo[j].drift = calcDisCPU(&newCentData[j*numDim],
                                           &oldCentData[j*numDim],
                                           numDim);

      if (newCentInfo[j].drift > newMaxDriftArr[newCentInfo[j].groupNum])
        {
          newMaxDriftArr[newCentInfo[j].groupNum] = newCentInfo[j].drift;
        }
  }
}

void calcWeightedMeansLloyd(CentInfo *newCentInfo,
                       CentInfo **allCentInfo,
                       DTYPE *newCentData,
                       DTYPE *oldCentData,
                       DTYPE **allCentData,
                       const int numCent,
                       const int numDim,
                       const int numGPU)
{
  DTYPE numerator = 0;
  DTYPE denominator = 0;
  DTYPE zeroNumerator = 0;
  int zeroCount = 0;

  for (int i = 0; i < numCent; i++)
  {
      for (int j = 0; j < numDim; j++)
      {
          oldCentData[(i * numDim) + j] = newCentData[(i * numDim) + j];
      }
  }

  for (int i = 0; i < numGPU; i++)
  {
      for (int j = 0; j < numCent; j++)
      {
        newCentInfo[j].count += allCentInfo[i][j].count;

        newCentInfo[j].groupNum = allCentInfo[0][j].groupNum;
      }
  }

  for (int j = 0; j < numCent; j++)
  {
      for (int k = 0; k < numDim; k++)
      {
          for (int l = 0; l < numGPU; l++)
          {
              if (allCentInfo[l][j].count == 0)
              {
                  zeroCount++;
                  zeroNumerator += allCentData[l][(j * numDim) + k];
              }

              numerator +=
              allCentData[l][(j * numDim) + k]*allCentInfo[l][j].count;

              denominator += allCentInfo[l][j].count;
          }

          if (denominator != 0)
          {
              newCentData[(j * numDim) + k] = numerator/denominator;
          }

          else
          {
              newCentData[(j * numDim) + k] = zeroNumerator/zeroCount;
          }

          zeroCount = 0;
          zeroNumerator = 0;
          numerator = 0;
          denominator = 0;
      }

			/*
      newCentInfo[j].drift = calcDisCPU(&newCentData[j*numDim],
                                           &oldCentData[j*numDim],
                                           numDim);
			*/
  }
}
