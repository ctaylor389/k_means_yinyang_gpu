#include "kmeansCPU.h"



/////////////////////////////////
// CPU k-means implementations //
/////////////////////////////////


double startFullOnCPU(PointInfo *pointInfo, 
                    CentInfo *centInfo, 
                    DTYPE *pointData, 
                    DTYPE *centData,
                    const int numPnt,
                    const int numCent,
                    const int numGrp, 
                    const int numDim,
                    const int numThread, 
                    const int maxIter, 
                    unsigned int *ranIter)
{
  double startTime = omp_get_wtime();

  // index variables
  unsigned int pntIndex, grpIndex;
  unsigned int index = 1;
  unsigned int conFlag = 0;

  // array to contain the maximum drift of each group of centroids
  // note: shared amongst all points
  DTYPE *maxDriftArr = (DTYPE *)malloc(sizeof(DTYPE) * numGrp);
  
  // array of all the points lower bounds
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numGrp);
  
  // initiatilize to INFINITY
  for(grpIndex = 0; grpIndex < numPnt * numGrp; grpIndex++)
  {
    pointLwrs[grpIndex] = INFINITY;
  }

  // array to contain integer flags which mark which groups need to be checked
  // for a potential new centroid
  // note: unique to each point
  unsigned int *groupLclArr = (unsigned int *)malloc(sizeof(unsigned int)*numPnt*numGrp);

  omp_set_num_threads(numThread);

  // the minimum of all the lower bounds for a single point
  DTYPE tmpGlobLwr = INFINITY;
  
  // cluster the centroids into NGROUPCPU groups
  groupCent(centInfo, centData, numCent, numGrp, numDim);

  // run one iteration of standard kmeans for initial centroid assignments
  initPoints(pointInfo, centInfo, pointData, pointLwrs, 
             centData, numPnt, numCent, numGrp, numDim, numThread);

  // master loop
  while(!conFlag && index < maxIter)
  {
    // clear drift array each new iteration
    for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
    {
      maxDriftArr[grpIndex] = 0.0;
    }

    // update centers via optimised update method
    updateCentroids(pointInfo, centInfo, pointData, centData, 
                    maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);

    // filtering done in parallel
    #pragma omp parallel \
    private(pntIndex, grpIndex, tmpGlobLwr) \
    shared(pointInfo, centInfo, pointData, centData, maxDriftArr, groupLclArr)
    {
      #pragma omp for schedule(static)
      for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
      {
        // reset old centroid before possibly finding a new one
        pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;

        tmpGlobLwr = INFINITY;
        
        // update upper bound
          // ub = ub + centroid's drift
        pointInfo[pntIndex].uprBound +=
          centInfo[pointInfo[pntIndex].centroidIndex].drift;
        
        // update group lower bounds
          // lb = lb - maxGroupDrift
        for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
        {
          pointLwrs[(pntIndex * numGrp) + grpIndex] -= maxDriftArr[grpIndex];

          if(pointLwrs[(pntIndex * numGrp) + grpIndex] < tmpGlobLwr)
          {
            // minimum lower bound
            tmpGlobLwr = pointLwrs[(pntIndex * numGrp) + grpIndex];
          }
        }

        // global filtering
        // if global lowerbound >= upper bound
        if(tmpGlobLwr < pointInfo[pntIndex].uprBound)
        {
          // tighten upperbound ub = d(x, b(x))
          pointInfo[pntIndex].uprBound = 
            calcDisCPU(&pointData[pntIndex * numDim],
                       &centData[pointInfo[pntIndex].centroidIndex * numDim],
                       numDim);
            
          // check condition again
          if(tmpGlobLwr < pointInfo[pntIndex].uprBound)
          {
            // group filtering
            for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
            {
                // mark groups that need to be checked
              if(pointLwrs[(pntIndex * numGrp) + grpIndex] < pointInfo[pntIndex].uprBound)
              groupLclArr[(pntIndex * numGrp) + grpIndex] = 1;
              else
              groupLclArr[(pntIndex * numGrp) + grpIndex] = 0;
            }
            // pass group array and point to go execute distance calculations
            pointCalcsFullCPU(&pointInfo[pntIndex], centInfo, 
                              &pointData[pntIndex*numDim], &pointLwrs[pntIndex*numGrp],
                              centData, maxDriftArr, &groupLclArr[pntIndex*numGrp], 
                              numPnt, numCent, numGrp, numDim);
          }
        }
      }
    }
    index++;
    conFlag = checkConverge(pointInfo, numPnt);
  }
  updateCentroids(pointInfo, centInfo, pointData, centData, 
                  maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);
  double endTime = omp_get_wtime();
  *ranIter = index;

  return endTime - startTime;
}



/*
CPU implementation of the simplified Yinyang algorithm
given only a file name with the points and a start and end time
*/
double startSimpleOnCPU(PointInfo *pointInfo, 
                      CentInfo *centInfo, 
                      DTYPE *pointData, 
                      DTYPE *centData,
                      const int numPnt,
                      const int numCent,
                      const int numGrp, 
                      const int numDim,
                      const int numThread,
                      const int maxIter, 
                      unsigned int *ranIter)
{
  double startTime = omp_get_wtime();

  // index variables
  unsigned int pntIndex, grpIndex;
  unsigned int index = 1;
  unsigned int conFlag = 0;

  // array to contain the maximum drift of each group of centroids
  // note: shared amongst all points
  DTYPE *maxDriftArr = (DTYPE *)malloc(sizeof(DTYPE) * numGrp);
  
  // array of all the points lower bounds
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt * numGrp);
  
  // initiatilize to INFINITY
  for(grpIndex = 0; grpIndex < numPnt * numGrp; grpIndex++)
  {
    pointLwrs[grpIndex] = INFINITY;
  }

  // array to contain integer flags which mark which groups need to be checked
  // for a potential new centroid
  // note: unique to each point
  unsigned int *groupLclArr = (unsigned int *)malloc(sizeof(unsigned int) * numPnt * numGrp);

  omp_set_num_threads(numThread);

  // the minimum of all the lower bounds for a single point
  DTYPE tmpGlobLwr = INFINITY;
  
  // cluster the centroids into NGROUPCPU groups
  groupCent(centInfo, centData, numCent, numGrp, numDim);

  // run one iteration of standard kmeans for initial centroid assignments
  initPoints(pointInfo, centInfo, pointData, pointLwrs, 
             centData, numPnt, numCent, numGrp, numDim, numThread);
  // master loop
  while(!conFlag && index < maxIter)
  {
    // clear drift array each new iteration
    for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
    {
      maxDriftArr[grpIndex] = 0.0;
    }
    // update centers via optimised update method
    updateCentroids(pointInfo, centInfo, pointData, centData, 
                    maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);
    // filtering done in parallel
    #pragma omp parallel \
    private(pntIndex, grpIndex, tmpGlobLwr) \
    shared(pointInfo, centInfo, pointData, centData, maxDriftArr, groupLclArr)
    {
      #pragma omp for schedule(static)
      for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
      {
        // reset old centroid before possibly finding a new one
        pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;

        tmpGlobLwr = INFINITY;

        // update upper bound
            // ub = ub + centroid's drift
        pointInfo[pntIndex].uprBound +=
          centInfo[pointInfo[pntIndex].centroidIndex].drift;
        
        // update group lower bounds
            // lb = lb - maxGroupDrift
        for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
        {
          pointLwrs[(pntIndex * numGrp) + grpIndex] -= maxDriftArr[grpIndex];

          if(pointLwrs[(pntIndex * numGrp) + grpIndex] < tmpGlobLwr)
          {
              // minimum lower bound
              tmpGlobLwr = pointLwrs[(pntIndex * numGrp) + grpIndex];
          }
        }

        // global filtering
        // if global lowerbound >= upper bound
        if(tmpGlobLwr < pointInfo[pntIndex].uprBound)
        {
          // tighten upperbound ub = d(x, b(x))
          pointInfo[pntIndex].uprBound = 
              calcDisCPU(&pointData[pntIndex * numDim],
                         &centData[pointInfo[pntIndex].centroidIndex * numDim],
                         numDim);
          // check condition again
          if(tmpGlobLwr < pointInfo[pntIndex].uprBound)
          {
            // group filtering
            for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
            {
              // mark groups that need to be checked
              if(pointLwrs[(pntIndex * numGrp) + grpIndex] < pointInfo[pntIndex].uprBound)
              groupLclArr[(pntIndex * numGrp) + grpIndex] = 1;
              else
              groupLclArr[(pntIndex * numGrp) + grpIndex] = 0;
            }
            
            // pass group array and point to go execute distance calculations
            pointCalcsSimpleCPU(&pointInfo[pntIndex], centInfo, 
                                &pointData[pntIndex*numDim], &pointLwrs[pntIndex*numGrp],
                                centData, maxDriftArr, &groupLclArr[pntIndex * numGrp], 
                                numPnt, numCent, numGrp, numDim);
          }
        }
      }
    }
    index++;
    conFlag = checkConverge(pointInfo, numPnt);
  }
  updateCentroids(pointInfo, centInfo, pointData, centData, 
                  maxDriftArr, numPnt, numCent, numGrp, numDim, numThread);
  double endTime = omp_get_wtime();
  *ranIter = index;

  return endTime - startTime;
}


double startSuperOnCPU(PointInfo *pointInfo, 
                     CentInfo *centInfo, 
                     DTYPE *pointData,
                     DTYPE *centData, 
                     const int numPnt, 
                     const int numCent, 
                     const int numDim,
                     const int numThread, 
                     const int maxIter, 
                     unsigned int *ranIter)
{
  double startTime = omp_get_wtime();

  // index variables
  unsigned int pntIndex, centIndex;
  unsigned int index = 1;

  DTYPE compDistance;
  DTYPE maxDrift = 0.0;

  unsigned int conFlag = 0;

  omp_set_num_threads(numThread);

  // place all centroids in one "group"
  for(centIndex = 0; centIndex < numCent; centIndex++)
  {
    centInfo[centIndex].groupNum = 0;
  }
  
  DTYPE *pointLwrs = (DTYPE *)malloc(sizeof(DTYPE) * numPnt);

  // run one iteration of standard kmeans for initial centroid assignments
  initPoints(pointInfo, centInfo, pointData, pointLwrs, 
             centData, numPnt, numCent, 1, numDim, numThread);

  // master loop
  while(!conFlag && index < maxIter)
  {
    // update centers via optimised update method
    updateCentroids(pointInfo, centInfo, pointData, centData, 
                    &maxDrift, numPnt, numCent, 1, numDim, numThread);
    
    // filtering done in parallel
    #pragma omp parallel \
    private(pntIndex, centIndex) \
    shared(pointInfo, centInfo, pointData, centData, maxDrift)
    {
      #pragma omp for schedule(static)
      for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
      {
        // reset old centroid before possibly finding a new one
        pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;

        // update upper bound
          // ub = ub + centroid's drift
        pointInfo[pntIndex].uprBound +=
          centInfo[pointInfo[pntIndex].centroidIndex].drift;

        // update lower bound
        pointLwrs[pntIndex] -= maxDrift;

        // global filtering
        // if global lowerbound >= upper bound
        if(pointLwrs[pntIndex] < pointInfo[pntIndex].uprBound)
        {
          // tighten upperbound ub = d(x, b(x))
          pointInfo[pntIndex].uprBound = 
            calcDisCPU(&pointData[pntIndex * numDim], 
                       &centData[pointInfo[pntIndex].centroidIndex * numDim],
                       numDim);

          // check condition again
          if(pointLwrs[pntIndex] < pointInfo[pntIndex].uprBound)
          {
            pointLwrs[pntIndex] = INFINITY;

            // calculate distance between point and every cluster
            for(centIndex = 0; centIndex < numCent; centIndex++)
            {
              // if clstIndex is the one already assigned, skip the calculation
              if(centIndex == pointInfo[pntIndex].oldCentroid)
              continue;

              compDistance = calcDisCPU(&pointData[pntIndex * numDim],
                                        &centData[centIndex * numDim], numDim);

              if(compDistance < pointInfo[pntIndex].uprBound)
              {
                pointLwrs[pntIndex] = pointInfo[pntIndex].uprBound;
                pointInfo[pntIndex].centroidIndex = centIndex;
                pointInfo[pntIndex].uprBound = compDistance;
              }
              else if(compDistance < pointLwrs[pntIndex])
              {
                pointLwrs[pntIndex] = compDistance;
              }
            }
          }
        }			
      }
    }
    index++;
    conFlag = checkConverge(pointInfo, numPnt);
  }
  updateCentroids(pointInfo, centInfo, pointData, centData, 
                  &maxDrift, numPnt, numCent, 1, numDim, numThread);
  double endTime = omp_get_wtime();
  *ranIter = index;

  return endTime - startTime;


}

double startLloydOnCPU(PointInfo *pointInfo, 
                     CentInfo *centInfo, 
                     DTYPE *pointData, 
                     DTYPE *centData, 
                     const int numPnt, 
                     const int numCent, 
                     const int numDim,
                     const int numThread,
                     const int maxIter, 
                     unsigned int *ranIter)
{

  double startTime = omp_get_wtime();
  
  unsigned int pntIndex, centIndex, dimIndex;
  unsigned int index = 0;
  unsigned int conFlag = 0;
  
  DTYPE *oldVecs = (DTYPE *)malloc(sizeof(DTYPE) * numDim * numCent);


  omp_set_num_threads(numThread);

  DTYPE currMin, currDis;

  // start standard kmeans algorithm for MAXITER iterations
  while(!conFlag && index < maxIter)
  {
    currMin = INFINITY;

    #pragma omp parallel \
    private(pntIndex, centIndex, currDis, currMin) \
    shared(pointInfo, centInfo, pointData, centData)
    {
        #pragma omp for nowait schedule(static)
        for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
        {
            pointInfo[pntIndex].oldCentroid = pointInfo[pntIndex].centroidIndex;
            for(centIndex = 0; centIndex < numCent; centIndex++)
            {
                currDis = calcDisCPU(&pointData[pntIndex * numDim],
                                     &centData[centIndex * numDim],
                                     numDim);
                if(currDis < currMin)
                {
                    pointInfo[pntIndex].centroidIndex = centIndex;
                    currMin = currDis;
                }
            }
            currMin = INFINITY;
        }
    }

    // clear centroids features
    for(centIndex = 0; centIndex < numCent; centIndex++)
    {
        for(dimIndex = 0; dimIndex < numDim; dimIndex++)
        {
            oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
            centData[(centIndex * numDim) + dimIndex] = 0.0;
        }
        centInfo[centIndex].count = 0;
    }
    // sum all assigned point's features
    #pragma omp parallel \
    private(pntIndex, dimIndex) \
    shared(pointInfo, centInfo, pointData, centData)
    {
      #pragma omp for nowait schedule(static)
      for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
      {
        #pragma omp atomic
        centInfo[pointInfo[pntIndex].centroidIndex].count++;

        for(dimIndex = 0; dimIndex < numDim; dimIndex++)
        {
          #pragma omp atomic
          centData[(pointInfo[pntIndex].centroidIndex * numDim) + dimIndex] += pointData[(pntIndex * numDim) + dimIndex];
        }
      }
    }
    // take the average of each feature to get new centroid features
    for(centIndex = 0; centIndex < numCent; centIndex++)
    {
      for(dimIndex = 0; dimIndex < numDim; dimIndex++)
      {
        if(centInfo[centIndex].count > 0)
        centData[(centIndex * numDim) + dimIndex] /= centInfo[centIndex].count;
        else
        centData[(centIndex * numDim) + dimIndex] = oldVecs[(centIndex * numDim) + dimIndex];
      }
    }
    index++;
    conFlag = checkConverge(pointInfo, numPnt);
  }

  *ranIter = index;

  double endTime = omp_get_wtime();
  
  free(oldVecs);
  
  return endTime - startTime;
}


unsigned int checkConverge(PointInfo *pointInfo, 
                           const int numPnt)
{
  for(int index = 0; index < numPnt; index++)
  {
    if(pointInfo[index].centroidIndex != pointInfo[index].oldCentroid)
    return 0;
  }
  return 1;
}



void pointCalcsFullCPU(PointInfo *pointInfoPtr,
                       CentInfo *centInfo,
                       DTYPE *pointDataPtr,
                       DTYPE *pointLwrPtr,
                       DTYPE *centData,
                       DTYPE *maxDriftArr,
                       unsigned int *groupArr,
                       const int numPnt,
                       const int numCent,
                       const int numGrp,
                       const int numDim)
{
  // index variables
  unsigned int grpIndex, centIndex;

  DTYPE compDistance;
  DTYPE oldLwr = INFINITY;
  DTYPE oldCentUpr = pointInfoPtr->uprBound;
  DTYPE oldCentLwr = pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum];

  for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
  {        
    if(groupArr[grpIndex])
    {
      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum)
      oldLwr = oldCentLwr + maxDriftArr[grpIndex];
      else
      oldLwr = pointLwrPtr[grpIndex] + maxDriftArr[grpIndex];

      // set group's lower bound to find new lower bound for this iteration
      pointLwrPtr[grpIndex] = INFINITY;

      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum && 
        pointInfoPtr->oldCentroid != pointInfoPtr->centroidIndex)
      pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum] = oldCentUpr;

      // loop through all of the group's centroids
      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        if(centIndex == pointInfoPtr->oldCentroid)
        continue;

        if(grpIndex == centInfo[centIndex].groupNum)
        {
          // local filtering condition
          if(pointLwrPtr[grpIndex] < oldLwr  - centInfo[centIndex].drift)
          continue;

          // perform distance calculation
          compDistance = 
            calcDisCPU(pointDataPtr, &centData[centIndex * numDim], numDim);

          if(compDistance < pointInfoPtr->uprBound)
          {
            pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] = pointInfoPtr->uprBound;
            pointInfoPtr->centroidIndex = centIndex;
            pointInfoPtr->uprBound = compDistance;
          }
          else if(compDistance < pointLwrPtr[grpIndex])
          {
            pointLwrPtr[grpIndex] = compDistance;
          }
        }
      }
    }
  }
}

void pointCalcsSimpleCPU(PointInfo *pointInfoPtr,
                       CentInfo *centInfo,
                       DTYPE *pointDataPtr,
                       DTYPE *pointLwrPtr,
                       DTYPE *centData,
                       DTYPE *maxDriftArr,
                       unsigned int *groupArr,
                       const int numPnt,
                       const int numCent,
                       const int numGrp,
                       const int numDim)
{
  // index variables
  unsigned int centIndex, grpIndex;

  DTYPE compDistance;

  for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
  {
    // if the group is not blocked by group filter
    if(groupArr[grpIndex])
    {
      // reset the lwrBoundArr to be only new lwrBounds
      pointLwrPtr[grpIndex] = INFINITY;
    }
  }
  
  for(centIndex = 0; centIndex < numCent; centIndex++)
  {
    // if the centroid's group is marked in groupArr
    if(groupArr[centInfo[centIndex].groupNum])
    {
      // if it was the originally assigned cluster, no need to calc dist
      if(centIndex == pointInfoPtr->oldCentroid)
      continue;
  
      // compute distance between point and centroid
      compDistance = calcDisCPU(pointDataPtr, &centData[centIndex * numDim], numDim);

      if(compDistance < pointInfoPtr->uprBound)
      {
        pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] = pointInfoPtr->uprBound;
        pointInfoPtr->centroidIndex = centIndex;
        pointInfoPtr->uprBound = compDistance;
      }
      else if(compDistance < pointLwrPtr[centInfo[centIndex].groupNum])
      {
        pointLwrPtr[centInfo[centIndex].groupNum] = compDistance;
      }
    }
  }
}



/*
Function used to do an intial iteration of K-means
*/
void initPoints(PointInfo *pointInfo,
              CentInfo *centInfo, 
              DTYPE *pointData, 
              DTYPE *pointLwrs, 
              DTYPE *centData, 
              const int numPnt, 
              const int numCent, 
              const int numGrp, 
              const int numDim, 
              const int numThread)
{
  unsigned int pntIndex, centIndex;

  DTYPE currDistance;
  
  // start single standard k-means iteration for initial bounds and cluster assignments
    // assignment
  #pragma omp parallel \
  private(pntIndex, centIndex, currDistance) \
  shared(pointInfo, centInfo, pointData, pointLwrs, centData)
  {
    #pragma omp for schedule(static)
    for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
    {
      pointInfo[pntIndex].uprBound = INFINITY;

      // for all centroids
      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        // currDistance is equal to the distance between the current feature
        // vector being inspected, and the current centroid being compared
        currDistance = calcDisCPU(&pointData[pntIndex * numDim], 
                                  &centData[centIndex * numDim],
                                  numDim);

        // if the the currDistance is less than the current minimum distance
        if(currDistance < pointInfo[pntIndex].uprBound)
        {
          if(pointInfo[pntIndex].uprBound != INFINITY)
          pointLwrs[(pntIndex * numGrp) + 
            centInfo[pointInfo[pntIndex].centroidIndex].groupNum] = 
              pointInfo[pntIndex].uprBound;
          // update assignment and upper bound
          pointInfo[pntIndex].centroidIndex = centIndex;
          pointInfo[pntIndex].uprBound = currDistance;
        }
        else if(currDistance < pointLwrs[(pntIndex * numGrp) + centInfo[centIndex].groupNum])
        {
          pointLwrs[(pntIndex * numGrp) + centInfo[centIndex].groupNum] = currDistance;
        }
      }
    }
  }
}

void updateCentroids(PointInfo *pointInfo, 
                   CentInfo *centInfo, 
                   DTYPE *pointData,
                   DTYPE *centData, 
                   DTYPE *maxDriftArr,
                   const int numPnt, 
                   const int numCent, 
                   const int numGrp, 
                   const int numDim, 
                   const int numThread)
{
  unsigned int pntIndex, centIndex, grpIndex, dimIndex;
  
  DTYPE compDrift;
  
  // holds the number of points assigned to each centroid formerly and currently
  unsigned int *oldCounts = (unsigned int *)malloc(sizeof(unsigned int) * numCent);
  unsigned int *newCounts = (unsigned int *)malloc(sizeof(unsigned int) * numCent);

  // holds the new vector calculated
  DTYPE *oldVecs = (DTYPE *)malloc(sizeof(DTYPE) * numCent * numDim);
  DTYPE oldCentFeat;


  omp_set_num_threads(numThread);

  omp_lock_t driftLock;
  omp_init_lock(&driftLock);
  
  // allocate data for new and old vector sums
  DTYPE *oldSums = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
  DTYPE *newSums = (DTYPE *)malloc(sizeof(DTYPE)*numCent*numDim);
  DTYPE oldSumFeat;
  DTYPE newSumFeat;

  for(centIndex = 0; centIndex < numCent; centIndex++)
  {
    for(dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
      oldSums[(centIndex * numDim) + dimIndex] = 0.0;
      newSums[(centIndex * numDim) + dimIndex] = 0.0;
    }
    oldCounts[centIndex] = 0;
    newCounts[centIndex] = 0;
  }
  for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
  {
    maxDriftArr[grpIndex] = 0.0;
  }
  
  for(pntIndex = 0; pntIndex < numPnt; pntIndex++)
  {
    // add one to the old count and new count for each centroid
    if(pointInfo[pntIndex].oldCentroid >= 0)
    oldCounts[pointInfo[pntIndex].oldCentroid]++;

    newCounts[pointInfo[pntIndex].centroidIndex]++;

    // if the old centroid does not match the new centroid,
    // add the points vector to each 
    if(pointInfo[pntIndex].oldCentroid != pointInfo[pntIndex].centroidIndex)
    {
      for(dimIndex = 0; dimIndex < numDim; dimIndex++)
      {
        if(pointInfo[pntIndex].oldCentroid >= 0)
        {
          oldSums[(pointInfo[pntIndex].oldCentroid * numDim) + dimIndex] += 
            pointData[(pntIndex * numDim) + dimIndex];
        }
        newSums[(pointInfo[pntIndex].centroidIndex * numDim) + dimIndex] += 
          pointData[(pntIndex * numDim) + dimIndex];
      }
    }
  }

  

  #pragma omp parallel \
  private(centIndex, dimIndex, oldCentFeat, oldSumFeat, newSumFeat, compDrift) \
  shared(driftLock, centInfo, centData, maxDriftArr, oldVecs)
  {
  // create new centroid points
    #pragma omp for schedule(static)
    for(centIndex = 0; centIndex < numCent; centIndex++)
    {
      for(dimIndex = 0; dimIndex < numDim; dimIndex++)
      {
        if(newCounts[centIndex] > 0)
        {
          oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
          oldCentFeat = oldVecs[(centIndex * numDim) + dimIndex];
          oldSumFeat = oldSums[(centIndex * numDim) + dimIndex];
          newSumFeat = newSums[(centIndex * numDim) + dimIndex];

          centData[(centIndex * numDim) + dimIndex] = 
          (oldCentFeat * oldCounts[centIndex] - oldSumFeat + newSumFeat) 
                                                  / newCounts[centIndex];
          //printf("(%f * %d - %f + %f) / %d\n", oldCentFeat,oldCounts[centIndex],oldSumFeat,newSumFeat,newCounts[centIndex]);
          
        }
        else
        {
          // if the centroid has no current members, no change occurs to its position
          oldVecs[(centIndex * numDim) + dimIndex] = centData[(centIndex * numDim) + dimIndex];
        }
      }
      compDrift = calcDisCPU(&oldVecs[centIndex * numDim], 
                             &centData[centIndex * numDim], numDim);
      omp_set_lock(&driftLock);
      // printf("%d\n",centInfo[centIndex].groupNum);
      if(compDrift > maxDriftArr[centInfo[centIndex].groupNum])
      {
        maxDriftArr[centInfo[centIndex].groupNum] = compDrift;
      }
      omp_unset_lock(&driftLock);
      centInfo[centIndex].drift = compDrift;
    }
  }
  omp_destroy_lock(&driftLock);
  
  free(oldCounts);
  free(newCounts);
  free(oldVecs);
  free(oldSums);
  free(newSums);
  
}



//////////////////////////////////////////////////
// Overloads for counting distance calculations //
//////////////////////////////////////////////////
