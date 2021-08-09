#include "kmeansKernel.h"

///////////////////////////////
// Atomic function overloads //
///////////////////////////////

/*
device helper function provides a double precision implementation
of atomicMax using atomicCAS
*/
__device__ void atomicMax(double *const address,
                        const double value)
{
  if (*address >= value)
  return;

  unsigned long long int * const address_as_i = (unsigned long long int *)address;
  unsigned long long int old = * address_as_i, assumed;

  do
  {
    assumed = old;
    if(__longlong_as_double(assumed) >= value)
    break;

    old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
  }while(assumed != old);
}

/*
device helper function provides a single precision implementation
of atomicMax using atomicCAS
*/
__device__ void atomicMax(float *const address,
                        const float value)
{
  if(*address >= value)
  return;

  int * const address_as_i = (int *)address;
  int old = * address_as_i, assumed;

  do
  {
    assumed = old;
    if(__int_as_float(assumed) >= value)
    break;

    old = atomicCAS(address_as_i, assumed, __float_as_int(value));
  } while(assumed != old);
}

//////////////////////////////
// Point Assignment Kernels //
//////////////////////////////

/*
Global kernel that assigns one thread to one point
Given points are each assigned a centroid and upper
and lower bounds
*/
__global__ void initRunKernel(PointInfo *pointInfo,
                            CentInfo *centInfo,
                            DTYPE *pointData,
                            DTYPE *pointLwrs,
                            DTYPE *centData,
                            const int numPnt,
                            const int numCent,
                            const int numGrp,
                            const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  unsigned int centIndex;

  DTYPE currDistance;
  pointInfo[tid].uprBound = INFINITY;

  for(centIndex = 0; centIndex < numCent; centIndex++)
  {
    // calculate euclidean distance between point and centroid
    currDistance = calcDis(&pointData[tid * numDim],
                           &centData[centIndex * numDim],
                           numDim);
    if(currDistance < pointInfo[tid].uprBound)
    {
      // make the former current min the new
      // lower bound for it's group
      if(pointInfo[tid].uprBound != INFINITY)
      pointLwrs[(tid * numGrp) + centInfo[pointInfo[tid].centroidIndex].groupNum] =
                                                                  pointInfo[tid].uprBound;

      // update assignment and upper bound
      pointInfo[tid].centroidIndex = centIndex;
      pointInfo[tid].uprBound = currDistance;
    }
    else if(currDistance < pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum])
    {
      pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum] = currDistance;
    }
  }
}

// Lloyds point assignment step
__global__ void assignPointsLloyd(PointInfo *pointInfo,
                                CentInfo *centInfo,
                                DTYPE *pointData,
                                DTYPE *centData,
                                const int numPnt,
                                const int numCent,
                                const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  DTYPE currMin = INFINITY;
  DTYPE currDis;

  unsigned int index;

  // reassign point's former centroid before finding new centroid
  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  for(index = 0; index < numCent; index++)
  {
    currDis = calcDis(&pointData[tid * numDim],
                      &centData[index * numDim],
                      numDim);
    if(currDis < currMin)
    {
      pointInfo[tid].centroidIndex = index;
      currMin = currDis;
    }
  }
}

/*
Full Yinyang algorithm point assignment step
Includes global, group, and local filters
*/
__global__ void assignPointsFull(PointInfo *pointInfo,
                               CentInfo *centInfo,
                               DTYPE *pointData,
                               DTYPE *pointLwrs,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  DTYPE tmpGlobLwr = INFINITY;

  int btid = threadIdx.x;
  unsigned int index;

  extern __shared__ unsigned int groupLclArr[];

  // reassign point's former centroid before finding new centroid
  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  // update points upper bound ub = ub + drift(b(x))
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

  // update group lower bounds
  // for all in lower bound array
  for(index = 0; index < numGrp; index++)
  {
    // subtract lowerbound by group's drift
    pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

    // if the lowerbound is less than the temp global lower,
    if(pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
    {
      // lower bound is new temp global lower
      tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
    }
  }

  // if the global lower bound is less than the upper bound
  if(tmpGlobLwr < pointInfo[tid].uprBound)
  {
    // tighten upper bound ub = d(x, b(x))
    pointInfo[tid].uprBound =
      calcDis(&pointData[tid * numDim],
              &centData[pointInfo[tid].centroidIndex * numDim], numDim);

    // if the lower bound is less than the upper bound
    if(tmpGlobLwr < pointInfo[tid].uprBound)
    {
      // loop through groups
      for(index = 0; index < numGrp; index++)
      {
        // if the lower bound is less than the upper bound
        // mark the group to go through the group filter
        if(pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
        groupLclArr[index + (btid * numGrp)] = 1;
        else
        groupLclArr[index + (btid * numGrp)] = 0;
      }

      // execute point calcs given the groups
      pointCalcsFull(&pointInfo[tid], centInfo, &pointData[tid * numDim],
                     &pointLwrs[tid * numGrp], centData, maxDriftArr,
                     &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim);
    }
  }
}
/*
Simplified Yinyang algorithm point assignment step
Includes global and group filters
*/
__global__ void assignPointsSimple(PointInfo *pointInfo,
                                 CentInfo *centInfo,
                                 DTYPE *pointData,
                                 DTYPE *pointLwrs,
                                 DTYPE *centData,
                                 DTYPE *maxDriftArr,
                                 const int numPnt,
                                 const int numCent,
                                 const int numGrp,
                                 const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  DTYPE tmpGlobLwr = INFINITY;

  unsigned int btid = threadIdx.x;
  unsigned int index;


  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  extern __shared__ unsigned int groupLclArr[];

  // update points upper bound
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

  // update group lower bounds
  // for all in lower bound array
  for(index = 0; index < numGrp; index++)
  {
    // subtract lowerbound by group's drift
    pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

    // if the lowerbound is less than the temp global lower,
    if(pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
    {
      // lower bound is new temp global lower
      tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
    }
  }

  // if the global lower bound is less than the upper bound
  if(tmpGlobLwr < pointInfo[tid].uprBound)
  {
    // tighten upper bound ub = d(x, b(x))
    pointInfo[tid].uprBound =
        calcDis(&pointData[tid * numDim],
                &centData[pointInfo[tid].centroidIndex * numDim],
                numDim);

    // if the lower bound is less than the upper bound
    if(tmpGlobLwr < pointInfo[tid].uprBound)
    {
      // loop through groups
      for(index = 0; index < numGrp; index++)
      {
        // if the lower bound is less than the upper bound
        // mark the group to go through the group filter
        if(pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
        groupLclArr[index + (btid * numGrp)] = 1;
        else
        groupLclArr[index + (btid * numGrp)] = 0;
      }

      // execute point calcs given the groups
      pointCalcsSimple(&pointInfo[tid],centInfo,&pointData[tid * numDim],
                       &pointLwrs[tid * numGrp], centData, maxDriftArr,
                       &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim);
    }

  }
}

/*
Super Simplified Yinyang algorithm point assignment step
Includes only the global filter
*/
__global__ void assignPointsSuper(PointInfo *pointInfo,
                                CentInfo *centInfo,
                                DTYPE *pointData,
                                DTYPE *pointLwrs,
                                DTYPE *centData,
                                DTYPE *maxDrift,
                                const int numPnt,
                                const int numCent,
                                const int numGrp,
                                const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  // point calc variables
  int centIndex;
  DTYPE compDistance;

  // set centroid's old centroid to be current assignment
  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  // update bounds
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;
  pointLwrs[tid * numGrp] -= *maxDrift;

  if(pointLwrs[tid * numGrp] < pointInfo[tid].uprBound)
  {
    // tighten upper bound
    pointInfo[tid].uprBound =
      calcDis(&pointData[tid * numDim],
              &centData[pointInfo[tid].centroidIndex * numDim],numDim);


    if(pointLwrs[(tid * numGrp)] < pointInfo[tid].uprBound)
    {
      // to get a new lower bound
      pointLwrs[tid * numGrp] = INFINITY;

      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        // do not calculate for the already assigned cluster
        if(centIndex == pointInfo[tid].oldCentroid)
        continue;

        compDistance = calcDis(&pointData[tid * numDim],
                               &centData[centIndex * numDim],
                               numDim);

        if(compDistance < pointInfo[tid].uprBound)
        {
          pointLwrs[tid * numGrp] = pointInfo[tid].uprBound;
          pointInfo[tid].centroidIndex = centIndex;
          pointInfo[tid].uprBound = compDistance;
        }
        else if(compDistance < pointLwrs[tid * numGrp])
        {
          pointLwrs[tid * numGrp] = compDistance;
        }
      }
    }
  }
}

////////////////////////////////////////
// Point Calculation Device Functions //
////////////////////////////////////////
__device__ void pointCalcsFull(PointInfo *pointInfoPtr,
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


  unsigned int grpIndex, centIndex;

  DTYPE compDistance;
  DTYPE oldLwr = INFINITY;
  DTYPE oldCentUpr = pointInfoPtr->uprBound;
  DTYPE oldCentLwr = pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum];

  // loop through all the groups
  for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
  {
    // if the group is marked as going through the group filter
    if(groupArr[grpIndex])
    {
      // save the former lower bound pre-update
      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum)
      oldLwr = oldCentLwr + maxDriftArr[grpIndex];
      else
      oldLwr = pointLwrPtr[grpIndex] + maxDriftArr[grpIndex];

      // reset the group's lower bound in order to find the new lower bound
      pointLwrPtr[grpIndex] = INFINITY;

      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum &&
        pointInfoPtr->oldCentroid != pointInfoPtr->centroidIndex)
      pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum] = oldCentUpr;

      // loop through all the group's centroids
      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        // if the cluster is the cluster already assigned
        // at the start of this iteration
        if(centIndex == pointInfoPtr->oldCentroid)
        continue;

        // if the cluster is a part of the group being checked now
        if(grpIndex == centInfo[centIndex].groupNum)
        {
          // local filtering condition
          if(pointLwrPtr[grpIndex] < oldLwr - centInfo[centIndex].drift)
          continue;

          // perform distance calculation
          compDistance = calcDis(pointDataPtr, &centData[centIndex * numDim], numDim);

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

__device__ void pointCalcsSimple(PointInfo *pointInfoPtr,
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

  unsigned int index;
  DTYPE compDistance;

  for(index = 0; index < numGrp; index++)
  {
    if(groupArr[index])
    {
      pointLwrPtr[index] = INFINITY;
    }
  }

  for(index = 0; index < numCent; index++)
  {
    if(groupArr[centInfo[index].groupNum])
    {
      if(index == pointInfoPtr->oldCentroid)
      continue;

      compDistance = calcDis(pointDataPtr,
                             &centData[index * numDim],
                             numDim);

      if(compDistance < pointInfoPtr->uprBound)
      {
        pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] =
          pointInfoPtr->uprBound;
        pointInfoPtr->centroidIndex = index;
        pointInfoPtr->uprBound = compDistance;
      }
      else if(compDistance < pointLwrPtr[centInfo[index].groupNum])
      {
        pointLwrPtr[centInfo[index].groupNum] = compDistance;
      }
    }
  }
}



//////////////////////////////////
// Centroid Calculation kernels //
//////////////////////////////////

__global__ void calcCentData(PointInfo *pointInfo,
                           CentInfo *centInfo,
                           DTYPE *pointData,
                           DTYPE *oldSums,
                           DTYPE *newSums,
                           unsigned int *oldCounts,
                           unsigned int *newCounts,
                           const int numPnt,
                           const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  unsigned int dimIndex;

  // atomicAdd 1 to old and new counts corresponding
  if(pointInfo[tid].oldCentroid >= 0)
  atomicAdd(&oldCounts[pointInfo[tid].oldCentroid], 1);

  atomicAdd(&newCounts[pointInfo[tid].centroidIndex], 1);

  // if old assignment and new assignment are not equal
  if(pointInfo[tid].oldCentroid != pointInfo[tid].centroidIndex)
  {
    // for all values in the vector
    for(dimIndex = 0; dimIndex < numDim; dimIndex++)
    {
      // atomic add the point's vector to the sum count
      if(pointInfo[tid].oldCentroid >= 0)
      {
        atomicAdd(&oldSums[(pointInfo[tid].oldCentroid * numDim) + dimIndex],
                  pointData[(tid * numDim) + dimIndex]);
      }
      atomicAdd(&newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex],
                pointData[(tid * numDim) + dimIndex]);
    }
  }

}

__global__ void calcNewCentroids(PointInfo *pointInfo,
                               CentInfo *centInfo,
                               DTYPE *centData,
                               DTYPE *oldCentData,
                               DTYPE *oldSums,
                               DTYPE *newSums,
                               DTYPE *maxDriftArr,
                               unsigned int *oldCounts,
                               unsigned int *newCounts,
                               const int numCent,
                               const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

  if(tid >= numCent)
  return;

  DTYPE oldFeature, oldSumFeat, newSumFeat, compDrift;

  unsigned int dimIndex;
  //vector oldVec;

  // create the new centroid vector
  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    if(newCounts[tid] > 0)
    {
      oldCentData[(tid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];

      oldFeature = centData[(tid * numDim) + dimIndex];
      oldSumFeat = oldSums[(tid * numDim) + dimIndex];
      newSumFeat = newSums[(tid * numDim) + dimIndex];

      centData[(tid * numDim) + dimIndex] =
        (oldFeature * oldCounts[tid] - oldSumFeat + newSumFeat)/newCounts[tid];
    }
    else
    {
      // no change to centroid
      oldCentData[(tid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];
    }
    newSums[(tid * numDim) + dimIndex] = 0.0;
    oldSums[(tid * numDim) + dimIndex] = 0.0;
  }


  // calculate the centroid's drift
  compDrift = calcDis(&oldCentData[tid * numDim],
                      &centData[tid * numDim],
                      numDim);

  atomicMax(&maxDriftArr[centInfo[tid].groupNum], compDrift);

  // set the centroid's vector to the new vector
  centInfo[tid].drift = compDrift;
  centInfo[tid].count = newCounts[tid];

  // clear the count and the sum arrays
  oldCounts[tid] = 0;
  newCounts[tid] = 0;

}


__global__ void calcCentDataLloyd(PointInfo *pointInfo,
                                DTYPE *pointData,
                                DTYPE *newSums,
                                unsigned int *newCounts,
                                const int numPnt,
                                const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

  if(tid >= numPnt)
  return;

  unsigned int dimIndex;

  // atomicAdd 1 to new counts corresponding
  atomicAdd(&newCounts[pointInfo[tid].centroidIndex], 1);

  // for all values in the vector
  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    atomicAdd(&newSums[(pointInfo[tid].centroidIndex * numDim) + dimIndex],
              pointData[(tid * numDim) + dimIndex]);
  }
}

__global__ void calcNewCentroidsLloyd(PointInfo *pointInfo,
                                    CentInfo *centInfo,
                                    DTYPE *centData,
                                    DTYPE *newSums,
                                    unsigned int *newCounts,
                                    const int numCent,
                                    const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numCent)
  return;

  unsigned int dimIndex;


  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    if(newCounts[tid] > 0)
    {
      centData[(tid * numDim) + dimIndex] =
        newSums[(tid * numDim) + dimIndex] / newCounts[tid];
    }
    // otherwise, no change
    newSums[(tid * numDim) + dimIndex] = 0.0;
  }
  newCounts[tid] = 0;
}

/*
this kernel is used to test performance differences between
the yinyang centroid update and the standard centroid update
*/
__global__ void calcNewCentroidsAve(PointInfo *pointInfo,
                                    CentInfo *centInfo,
                                    DTYPE *centData,
                                    DTYPE *newSums,
                                    DTYPE *maxDriftArr,
                                    unsigned int *newCounts,
                                    const int numCent,
                                    const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  unsigned int btid = threadIdx.x;

  if(tid >= numCent)
  return;

  unsigned int dimIndex;

  DTYPE compDrift;

  extern __shared__ DTYPE oldCentPos[];

  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    if(newCounts[tid] > 0)
    {
      oldCentPos[(btid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];

      centData[(tid * numDim) + dimIndex] =
        newSums[(tid * numDim) + dimIndex] / newCounts[tid];

      newSums[(tid * numDim) + dimIndex] = 0.0;
    }
    else
    {
      oldCentPos[(btid * numDim) + dimIndex] = centData[(tid * numDim) + dimIndex];

      newSums[(tid * numDim) + dimIndex] = 0.0;
    }
  }

  // compute drift
  compDrift = calcDis(&oldCentPos[btid * numDim],
                      &centData[tid * numDim], numDim);
  centInfo[tid].drift = compDrift;

  atomicMax(&maxDriftArr[centInfo[tid].groupNum], compDrift);

  newCounts[tid] = 0;
}


////////////////////
// Helper Kernels //
////////////////////

// warms up gpu for time trialing
__global__ void warmup(unsigned int * tmp)
{
  if(threadIdx.x == 0)
  {
      *tmp = 555;
  }
  return;
}

// checks convergence of data on GPU
__global__ void checkConverge(PointInfo *pointInfo,
                              unsigned int *conFlag,
                              const int numPnt)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  if(pointInfo[tid].oldCentroid != pointInfo[tid].centroidIndex)
  atomicCAS(conFlag, 0, 1);

}

/*
simple helper kernel that clears the drift array of size T
on the GPU. Called once each iteration for a total of MAXITER times
*/
__global__ void clearDriftArr(DTYPE *maxDriftArr,
                              const int numGrp)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numGrp)
  return;

  maxDriftArr[tid] = 0.0;
}

__global__ void clearCentCalcData(DTYPE *newCentSum,
                                  DTYPE *oldCentSum,
                                  unsigned int *newCentCount,
                                  unsigned int *oldCentCount,
                                  const int numCent,
                                  const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numCent)
  return;

  unsigned int dimIndex;

  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    newCentSum[(tid * numDim) + dimIndex] = 0.0;
    oldCentSum[(tid * numDim) + dimIndex] = 0.0;
  }
  newCentCount[tid] = 0;
  oldCentCount[tid] = 0;

}

__global__ void clearCentCalcDataLloyd(DTYPE *newCentSum,
                                     unsigned int *newCentCount,
                                     const int numCent,
                                     const int numDim)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numCent)
  return;

  unsigned int dimIndex;

  for(dimIndex = 0; dimIndex < numDim; dimIndex++)
  {
    newCentSum[(tid * numDim) + dimIndex] = 0.0;
  }
  newCentCount[tid] = 0;
}

/////////////////////////////
// device Helper Functions //
/////////////////////////////

/*
Simple device helper function that takes in two vectors and returns
the euclidean distance between them at DTYPE precision
*/
__device__ DTYPE calcDis(DTYPE *vec1, DTYPE *vec2, const int numDim)
{
  int index;
  DTYPE total = 0;
  DTYPE square;

  for(index = 0; index < numDim; index++)
  {
    square = (vec1[index] - vec2[index]);
    total += square * square;
  }

  return sqrt(total);
}



/////////////////////////////////////////////////////////////////////////
// Overloaded kernels and functions for counting distance calculations //
/////////////////////////////////////////////////////////////////////////

/*
Global kernel that assigns one thread to one point
Given points are each assigned a centroid and upper
and lower bounds
*/
__global__ void initRunKernel(PointInfo *pointInfo,
                            CentInfo *centInfo,
                            DTYPE *pointData,
                            DTYPE *pointLwrs,
                            DTYPE *centData,
                            const int numPnt,
                            const int numCent,
                            const int numGrp,
                            const int numDim,
                            unsigned long long int *calcCount)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  unsigned int centIndex;

  DTYPE currDistance;
  pointInfo[tid].uprBound = INFINITY;

  for(centIndex = 0; centIndex < numCent; centIndex++)
  {
    // calculate euclidean distance between point and centroid
    currDistance = calcDis(&pointData[tid * numDim],
                           &centData[centIndex * numDim],
                           numDim);
    atomicAdd(calcCount, 1);
    if(currDistance < pointInfo[tid].uprBound)
    {
      // make the former current min the new
      // lower bound for it's group
      if(pointInfo[tid].uprBound != INFINITY)
      pointLwrs[(tid * numGrp) + centInfo[pointInfo[tid].centroidIndex].groupNum] =
                                                                  pointInfo[tid].uprBound;

      // update assignment and upper bound
      pointInfo[tid].centroidIndex = centIndex;
      pointInfo[tid].uprBound = currDistance;
    }
    else if(currDistance < pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum])
    {
      pointLwrs[(tid * numGrp) + centInfo[centIndex].groupNum] = currDistance;
    }
  }
}


/*
Full Yinyang algorithm point assignment step
Includes global, group, and local filters
*/
__global__ void assignPointsFull(PointInfo *pointInfo,
                               CentInfo *centInfo,
                               DTYPE *pointData,
                               DTYPE *pointLwrs,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim,
                               unsigned long long int *calcCount)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  DTYPE tmpGlobLwr = INFINITY;

  int btid = threadIdx.x;
  unsigned int index;

  extern __shared__ unsigned int groupLclArr[];

  // reassign point's former centroid before finding new centroid
  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  // update points upper bound ub = ub + drift(b(x))
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

  // update group lower bounds
  // for all in lower bound array
  for(index = 0; index < numGrp; index++)
  {
    // subtract lowerbound by group's drift
    pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

    // if the lowerbound is less than the temp global lower,
    if(pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
    {
      // lower bound is new temp global lower
      tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
    }
  }

  // if the global lower bound is less than the upper bound
  if(tmpGlobLwr < pointInfo[tid].uprBound)
  {
    // tighten upper bound ub = d(x, b(x))
    pointInfo[tid].uprBound =
      calcDis(&pointData[tid * numDim],
              &centData[pointInfo[tid].centroidIndex * numDim], numDim);
    atomicAdd(calcCount, 1);
    // if the lower bound is less than the upper bound
    if(tmpGlobLwr < pointInfo[tid].uprBound)
    {
      // loop through groups
      for(index = 0; index < numGrp; index++)
      {
        // if the lower bound is less than the upper bound
        // mark the group to go through the group filter
        if(pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
        groupLclArr[index + (btid * numGrp)] = 1;
        else
        groupLclArr[index + (btid * numGrp)] = 0;
      }

      // execute point calcs given the groups
      pointCalcsFull(&pointInfo[tid], centInfo, &pointData[tid * numDim],
                     &pointLwrs[tid * numGrp], centData, maxDriftArr,
                     &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim);
    }
  }
}
/*
Simplified Yinyang algorithm point assignment step
Includes global and group filters
*/
__global__ void assignPointsSimple(PointInfo *pointInfo,
                                 CentInfo *centInfo,
                                 DTYPE *pointData,
                                 DTYPE *pointLwrs,
                                 DTYPE *centData,
                                 DTYPE *maxDriftArr,
                                 const int numPnt,
                                 const int numCent,
                                 const int numGrp,
                                 const int numDim,
                                 unsigned long long int *calcCount)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  DTYPE tmpGlobLwr = INFINITY;

  unsigned int btid = threadIdx.x;
  unsigned int index;


  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  extern __shared__ unsigned int groupLclArr[];

  // update points upper bound
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;

  // update group lower bounds
  // for all in lower bound array
  for(index = 0; index < numGrp; index++)
  {
    // subtract lowerbound by group's drift
    pointLwrs[(tid * numGrp) + index] -= maxDriftArr[index];

    // if the lowerbound is less than the temp global lower,
    if(pointLwrs[(tid * numGrp) + index] < tmpGlobLwr)
    {
      // lower bound is new temp global lower
      tmpGlobLwr = pointLwrs[(tid * numGrp) + index];
    }
  }

  // if the global lower bound is less than the upper bound
  if(tmpGlobLwr < pointInfo[tid].uprBound)
  {
    // tighten upper bound ub = d(x, b(x))
    pointInfo[tid].uprBound =
        calcDis(&pointData[tid * numDim],
                &centData[pointInfo[tid].centroidIndex * numDim],
                numDim);
    atomicAdd(calcCount, 1);

    // if the lower bound is less than the upper bound
    if(tmpGlobLwr < pointInfo[tid].uprBound)
    {
      // loop through groups
      for(index = 0; index < numGrp; index++)
      {
        // if the lower bound is less than the upper bound
        // mark the group to go through the group filter
        if(pointLwrs[(tid * numGrp) + index] < pointInfo[tid].uprBound)
        groupLclArr[index + (btid * numGrp)] = 1;
        else
        groupLclArr[index + (btid * numGrp)] = 0;
      }

      // execute point calcs given the groups
      pointCalcsSimple(&pointInfo[tid],centInfo,&pointData[tid * numDim],
                       &pointLwrs[tid * numGrp], centData, maxDriftArr,
                       &groupLclArr[btid * numGrp], numPnt, numCent, numGrp, numDim, calcCount);
    }

  }
}

/*
Super Simplified Yinyang algorithm point assignment step
Includes only the global filter
*/
__global__ void assignPointsSuper(PointInfo *pointInfo,
                                CentInfo *centInfo,
                                DTYPE *pointData,
                                DTYPE *pointLwrs,
                                DTYPE *centData,
                                DTYPE *maxDrift,
                                const int numPnt,
                                const int numCent,
                                const int numGrp,
                                const int numDim,
                                unsigned long long int *calcCount)
{
  unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
  if(tid >= numPnt)
  return;

  // point calc variables
  int centIndex;
  DTYPE compDistance;

  // set centroid's old centroid to be current assignment
  pointInfo[tid].oldCentroid = pointInfo[tid].centroidIndex;

  // update bounds
  pointInfo[tid].uprBound += centInfo[pointInfo[tid].centroidIndex].drift;
  pointLwrs[tid * numGrp] -= *maxDrift;

  if(pointLwrs[tid * numGrp] < pointInfo[tid].uprBound)
  {
    // tighten upper bound
    pointInfo[tid].uprBound =
      calcDis(&pointData[tid * numDim],
              &centData[pointInfo[tid].centroidIndex * numDim],numDim);
    atomicAdd(calcCount, 1);

    if(pointLwrs[(tid * numGrp)] < pointInfo[tid].uprBound)
    {
      // to get a new lower bound
      pointLwrs[tid * numGrp] = INFINITY;

      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        // do not calculate for the already assigned cluster
        if(centIndex == pointInfo[tid].oldCentroid)
        continue;

        compDistance = calcDis(&pointData[tid * numDim],
                               &centData[centIndex * numDim],
                               numDim);

        atomicAdd(calcCount, 1);

        if(compDistance < pointInfo[tid].uprBound)
        {
          pointLwrs[tid * numGrp] = pointInfo[tid].uprBound;
          pointInfo[tid].centroidIndex = centIndex;
          pointInfo[tid].uprBound = compDistance;
        }
        else if(compDistance < pointLwrs[tid * numGrp])
        {
          pointLwrs[tid * numGrp] = compDistance;
        }
      }
    }
  }
}

////////////////////////////////////////
// Point Calculation Device Functions //
////////////////////////////////////////
__device__ void pointCalcsFull(PointInfo *pointInfoPtr,
                             CentInfo *centInfo,
                             DTYPE *pointDataPtr,
                             DTYPE *pointLwrPtr,
                             DTYPE *centData,
                             DTYPE *maxDriftArr,
                             unsigned int *groupArr,
                             const int numPnt,
                             const int numCent,
                             const int numGrp,
                             const int numDim,
                             unsigned long long int *calcCount)
{


  unsigned int grpIndex, centIndex;

  DTYPE compDistance;
  DTYPE oldLwr = INFINITY;
  DTYPE oldCentUpr = pointInfoPtr->uprBound;
  DTYPE oldCentLwr = pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum];

  // loop through all the groups
  for(grpIndex = 0; grpIndex < numGrp; grpIndex++)
  {
    // if the group is marked as going through the group filter
    if(groupArr[grpIndex])
    {
      // save the former lower bound pre-update
      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum)
      oldLwr = oldCentLwr + maxDriftArr[grpIndex];
      else
      oldLwr = pointLwrPtr[grpIndex] + maxDriftArr[grpIndex];

      // reset the group's lower bound in order to find the new lower bound
      pointLwrPtr[grpIndex] = INFINITY;

      if(grpIndex == centInfo[pointInfoPtr->oldCentroid].groupNum &&
        pointInfoPtr->oldCentroid != pointInfoPtr->centroidIndex)
      pointLwrPtr[centInfo[pointInfoPtr->oldCentroid].groupNum] = oldCentUpr;

      // loop through all the group's centroids
      for(centIndex = 0; centIndex < numCent; centIndex++)
      {
        // if the cluster is the cluster already assigned
        // at the start of this iteration
        if(centIndex == pointInfoPtr->oldCentroid)
        continue;

        // if the cluster is a part of the group being checked now
        if(grpIndex == centInfo[centIndex].groupNum)
        {
          // local filtering condition
          if(pointLwrPtr[grpIndex] < oldLwr - centInfo[centIndex].drift)
          continue;

          // perform distance calculation
          compDistance = calcDis(pointDataPtr, &centData[centIndex * numDim], numDim);
          atomicAdd(calcCount, 1);
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

__device__ void pointCalcsSimple(PointInfo *pointInfoPtr,
                               CentInfo *centInfo,
                               DTYPE *pointDataPtr,
                               DTYPE *pointLwrPtr,
                               DTYPE *centData,
                               DTYPE *maxDriftArr,
                               unsigned int *groupArr,
                               const int numPnt,
                               const int numCent,
                               const int numGrp,
                               const int numDim,
                               unsigned long long int *calcCount)
{

  unsigned int index;
  DTYPE compDistance;

  for(index = 0; index < numGrp; index++)
  {
    if(groupArr[index])
    {
      pointLwrPtr[index] = INFINITY;
    }
  }

  for(index = 0; index < numCent; index++)
  {
    if(groupArr[centInfo[index].groupNum])
    {
      if(index == pointInfoPtr->oldCentroid)
      continue;

      compDistance = calcDis(pointDataPtr,
                             &centData[index * numDim],
                             numDim);
      atomicAdd(calcCount, 1);

      if(compDistance < pointInfoPtr->uprBound)
      {
        pointLwrPtr[centInfo[pointInfoPtr->centroidIndex].groupNum] =
          pointInfoPtr->uprBound;
        pointInfoPtr->centroidIndex = index;
        pointInfoPtr->uprBound = compDistance;
      }
      else if(compDistance < pointLwrPtr[centInfo[index].groupNum])
      {
        pointLwrPtr[centInfo[index].groupNum] = compDistance;
      }
    }
  }
}

