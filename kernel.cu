#include "kernel.h"

///////////////////////////////
// Atomic function overloads //
///////////////////////////////

/*
device helper function provides a double precision implementation 
of atomicMax using atomicCAS 
*/
__device__ void atomicMax(double *const address, const double value)
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
__device__ void atomicMax(float *const address, const float value)
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
__global__ void initRunKernel(point * dataset,
                              cent * centroidDataset)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    unsigned int centIndex;

    DTYPE currDistance;
    dataset[tid].uprBound = INFINITY;

    for(centIndex = 0; centIndex < NCLUST; centIndex++)
    {
        // calculate euclidean distance between point and centroid
        currDistance = calcDis(&dataset[tid].vec, 
                               &centroidDataset[centIndex].vec);
        if(currDistance < dataset[tid].uprBound)
        {
            // make the former current min the new 
            // lower bound for it's group
            if(dataset[tid].uprBound != INFINITY)
            dataset[tid].lwrBoundArr[centroidDataset[dataset[tid].centroidIndex].groupNum] = dataset[tid].uprBound;

            // update assignment and upper bound
            dataset[tid].centroidIndex = centIndex;
            dataset[tid].uprBound = currDistance;
        }
        else if(currDistance < dataset[tid].lwrBoundArr[centroidDataset[centIndex].groupNum])
        {
            dataset[tid].lwrBoundArr[centroidDataset[centIndex].groupNum] = currDistance;
        }
    }
}

// Lloyds point assignment step
__global__ void assignPointsLloyd(point * dataset,
                                  cent * centroidDataset)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    DTYPE currMin = INFINITY;
    DTYPE currDis;

    unsigned int index;

    // reassign point's former centroid before finding new centroid
    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    for(index = 0; index < NCLUST; index++)
    {
        currDis = calcDis(&dataset[tid].vec, 
                          &centroidDataset[index].vec);
        if(currDis < currMin)
        {
            dataset[tid].centroidIndex = index;
            currMin = currDis;
        }
    }
}

/*
Full Yinyang algorithm point assignment step
Includes global, group, and local filters
*/
__global__ void assignPointsFull(point *dataset,
                                 cent *centroidDataset,
                                 DTYPE *maxDriftArr)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    DTYPE tmpGlobLwr = INFINITY;

    unsigned int btid = threadIdx.x;
    unsigned int index;

    __shared__ unsigned int groupLclArr[NGROUP*BLOCKSIZE];

    // reassign point's former centroid before finding new centroid
    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    // update points upper bound ub = ub + drift(b(x))
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for(index = 0; index < NGROUP; index++)
    {        
        // subtract lowerbound by group's drift
        dataset[tid].lwrBoundArr[index] -= maxDriftArr[index];

        // if the lowerbound is less than the temp global lower,
        if(dataset[tid].lwrBoundArr[index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = dataset[tid].lwrBoundArr[index];
        }
    }

    // if the global lower bound is less than the upper bound
    if(tmpGlobLwr < dataset[tid].uprBound)
    {
        // tighten upper bound ub = d(x, b(x))
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                &centroidDataset[dataset[tid].centroidIndex].vec);

        // if the lower bound is less than the upper bound
        if(tmpGlobLwr < dataset[tid].uprBound)
        {
            // loop through groups
            for(index = 0; index < NGROUP; index++)
            {
                // if the lower bound is less than the upper bound
                // mark the group to go through the group filter
                if(dataset[tid].lwrBoundArr[index] < dataset[tid].uprBound)
                groupLclArr[index + (btid * NGROUP)] = 1;
                else
                groupLclArr[index + (btid * NGROUP)] = 0;
            }

            // execute point calcs given the groups
            pointCalcsFullAlt(&dataset[tid],
                           centroidDataset,
                           &groupLclArr[btid * NGROUP],
                           maxDriftArr);
        }
    }
}
/*
Simplified Yinyang algorithm point assignment step
Includes global and group filters
*/
__global__ void assignPointsSimple(point *dataset,
                                   cent *centroidDataset,
                                   DTYPE *maxDriftArr)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    DTYPE tmpGlobLwr = INFINITY;

    unsigned int btid = threadIdx.x;
    unsigned int index;


    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    __shared__ unsigned int groupLclArr[NGROUP*BLOCKSIZE];

    // update points upper bound
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for(index = 0; index < NGROUP; index++)
    {
        // subtract lowerbound by group's drift
        dataset[tid].lwrBoundArr[index] -= maxDriftArr[index];

        // if the lowerbound is less than the temp global lower,
        if(dataset[tid].lwrBoundArr[index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = dataset[tid].lwrBoundArr[index];
        }
    }

    // if the global lower bound is less than the upper bound
    if(tmpGlobLwr < dataset[tid].uprBound)
    {
        // tighten upper bound ub = d(x, b(x))
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                &centroidDataset[dataset[tid].centroidIndex].vec);

        // if the lower bound is less than the upper bound
        if(tmpGlobLwr < dataset[tid].uprBound)
        {
            // loop through groups
            for(index = 0; index < NGROUP; index++)
            {
                // if the lower bound is less than the upper bound
                // mark the group to go through the group filter
                if(dataset[tid].lwrBoundArr[index] < dataset[tid].uprBound)
                groupLclArr[index + (btid * NGROUP)] = 1;
                else
                groupLclArr[index + (btid * NGROUP)] = 0;
            }

            // execute point calcs given the groups
            pointCalcsSimple(&dataset[tid],
                             centroidDataset,
                             &groupLclArr[btid * NGROUP],
                             maxDriftArr);
        }

    }
}

/*
Super Simplified Yinyang algorithm point assignment step
Includes only the global filter
*/
__global__ void assignPointsSuper(point *dataset,
                                  cent *centroidDataset,
                                  DTYPE *maxDrift)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    // point calc variables
    unsigned int clustIndex;
    DTYPE compDistance;

    // set centroid's old centroid to be current assignment
    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    // update bounds
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;
    dataset[tid].lwrBoundArr[0] -= *maxDrift;

    if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
    {
        // tighten upper bound
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                    &centroidDataset[dataset[tid].centroidIndex].vec);


        if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
        {
            // to get a new lower bound
            dataset[tid].lwrBoundArr[0] = INFINITY;

            for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
            {
                // do not calculate for the already assigned cluster
                if(clustIndex == dataset[tid].oldCentroid)
                continue;

                compDistance = calcDis(&dataset[tid].vec,
                                       &centroidDataset[clustIndex].vec);

                if(compDistance < dataset[tid].uprBound)
                {
                    dataset[tid].lwrBoundArr[0] = dataset[tid].uprBound;
                    dataset[tid].centroidIndex = clustIndex;
                    dataset[tid].uprBound = compDistance;
                }
                else if(compDistance < dataset[tid].lwrBoundArr[0])
                {
                    dataset[tid].lwrBoundArr[0] = compDistance;
                }
            }
        }
    }
}

////////////////////////////////////////
// Point Calculation Device Functions //
////////////////////////////////////////
__device__ void pointCalcsFull(point *pointPtr,
                               cent *centroidDataset,
                               unsigned int *groupArr,
                               DTYPE *maxDriftArr)
{


    unsigned int grpIndex, clstIndex;

    DTYPE compDistance;
    DTYPE oldLwr = INFINITY;
    DTYPE oldCentUpr = pointPtr->uprBound;
    DTYPE oldCentLwr = pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum];

    // loop through all the groups 
    for(grpIndex = 0; grpIndex < NGROUP; grpIndex++)
    {
        // if the group is marked as going through the group filter
        if(groupArr[grpIndex])
        {
            // save the former lower bound pre-update
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum)
            oldLwr = oldCentLwr + maxDriftArr[grpIndex];
            else
            oldLwr = pointPtr->lwrBoundArr[grpIndex] + maxDriftArr[grpIndex];

            // reset the group's lower bound in order to find the new lower bound
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;

            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;

            // loop through all the group's clusters
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                // if the cluster is the cluster already assigned 
                // at the start of this iteration
                if(clstIndex == pointPtr->oldCentroid)
                continue;

                // if the cluster is a part of the group being checked now
                if(grpIndex == centroidDataset[clstIndex].groupNum)
                {   
                    // local filtering condition
                    if(pointPtr->lwrBoundArr[grpIndex] < oldLwr - centroidDataset[clstIndex].drift)
                    continue;

                    // perform distance calculation
                    compDistance = calcDis(&pointPtr->vec, &centroidDataset[clstIndex].vec);

                    if(compDistance < pointPtr->uprBound)
                    {
                        pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                        pointPtr->centroidIndex = clstIndex;
                        pointPtr->uprBound = compDistance;
                    }
                    else if(compDistance < pointPtr->lwrBoundArr[grpIndex])
                    {
                        pointPtr->lwrBoundArr[grpIndex] = compDistance;
                    }
                }
            }
        }
    }
}

__device__ void pointCalcsSimple(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr)
{

    unsigned int index;
    DTYPE compDistance;

    for(index = 0; index < NGROUP; index++)
    {
        if(groupArr[index])
        {
            pointPtr->lwrBoundArr[index] = INFINITY;
        }
    }

    for(index = 0; index < NCLUST; index++)
    {
        if(groupArr[centroidDataset[index].groupNum])
        {
            if(index == pointPtr->oldCentroid)
            continue;

            compDistance = calcDis(&pointPtr->vec, 
                                   &centroidDataset[index].vec);

            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = index;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[index].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[index].groupNum] = compDistance;
            }
        }
    }
}


__device__ void pointCalcsFullAlt(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr)
{
    unsigned int index;
    DTYPE compDistance;
    DTYPE oldLwrs[NGROUP];

    for(index = 0; index < NGROUP; index++)
    {
        if(groupArr[index])
        {
            oldLwrs[index] = pointPtr->lwrBoundArr[index] + maxDriftArr[index];
            pointPtr->lwrBoundArr[index] = INFINITY;
        }
    }

    for(index = 0; index < NCLUST; index++)
    {
        if(groupArr[centroidDataset[index].groupNum])
        {
            if(index == pointPtr->oldCentroid)
            continue;

            if(pointPtr->lwrBoundArr[centroidDataset[index].groupNum] < oldLwrs[centroidDataset[index].groupNum] - centroidDataset[index].drift)
            continue;

            compDistance = calcDis(&pointPtr->vec,
                                   &centroidDataset[index].vec);

            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = index;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[index].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[index].groupNum] = compDistance;
            }
        }
    }
}






//////////////////////////////////
// Centroid Calculation kernels //
//////////////////////////////////

__global__ void calcCentData(point *dataset,
                        cent *centroidDataset,
                        vector *oldSums,
                        vector *newSums,
                        unsigned int *oldCounts,
                        unsigned int *newCounts)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    unsigned int dimIndex;

    // atomicAdd 1 to old and new counts corresponding
    if(dataset[tid].oldCentroid >= 0)
    atomicAdd(&oldCounts[dataset[tid].oldCentroid], 1);

    atomicAdd(&newCounts[dataset[tid].centroidIndex], 1);

    // if old assignment and new assignment are not equal
    if(dataset[tid].oldCentroid != dataset[tid].centroidIndex)
    {
        // for all values in the vector
        for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
        {
            // atomic add the point's vector to the sum count
            if(dataset[tid].oldCentroid >= 0)
            {
                atomicAdd(&oldSums[dataset[tid].oldCentroid].feat[dimIndex],
                    dataset[tid].vec.feat[dimIndex]);
            }	
            atomicAdd(&newSums[dataset[tid].centroidIndex].feat[dimIndex],
                dataset[tid].vec.feat[dimIndex]);
        }
    }
    
}

__global__ void calcNewCentroids(point *dataset,
                        cent *centroidDataset,
                        DTYPE *maxDriftArr,
                        vector *oldSums,
                        vector *newSums,
                        unsigned int *oldCounts,
                        unsigned int *newCounts)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

    if(tid >= NCLUST)
    return;

    DTYPE oldFeature, oldSumFeat, newSumFeat, compDrift;
    
    unsigned int dimIndex;

    vector oldVec;

    // create the new centroid vector
    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        if(newCounts[tid] > 0)
        {
            oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
            oldFeature = centroidDataset[tid].vec.feat[dimIndex];
            oldSumFeat = oldSums[tid].feat[dimIndex];
            newSumFeat = newSums[tid].feat[dimIndex];

            centroidDataset[tid].vec.feat[dimIndex] = 
                (oldFeature * oldCounts[tid] - oldSumFeat + newSumFeat)
                                                            / newCounts[tid];			

            newSums[tid].feat[dimIndex] = 0.0;
            oldSums[tid].feat[dimIndex] = 0.0;
        }
        else
        {
            // no change to centroid
            oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
            newSums[tid].feat[dimIndex] = 0.0;
            oldSums[tid].feat[dimIndex] = 0.0;
        }
    }
    

    // calculate the centroid's drift
    compDrift = calcDis(&oldVec, &centroidDataset[tid].vec);

    atomicMax(&maxDriftArr[centroidDataset[tid].groupNum], compDrift);

    // set the centroid's vector to the new vector
    centroidDataset[tid].drift = compDrift;


    // clear the count and the sum arrays
    oldCounts[tid] = 0;
    newCounts[tid] = 0;
}


__global__ void calcCentDataLloyd(point *dataset,
                                  cent *centroidDataset,
                                  vector *newSums,
                                  unsigned int *newCounts)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

    if(tid >= NPOINT)
    return;

    unsigned int dimIndex;

    // atomicAdd 1 to new counts corresponding
    atomicAdd(&newCounts[dataset[tid].centroidIndex], 1);

    // for all values in the vector
    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        atomicAdd(&newSums[dataset[tid].centroidIndex].feat[dimIndex],
            dataset[tid].vec.feat[dimIndex]);
    }
}

__global__ void calcNewCentroidsLloyd(point *dataset,
                                      cent *centroidDataset,
                                      vector *newSums,
                                      unsigned int *newCounts)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NCLUST)
    return;

    unsigned int dimIndex;


    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        if(newCounts[tid] > 0)
        {
            centroidDataset[tid].vec.feat[dimIndex] =
                newSums[tid].feat[dimIndex] / newCounts[tid];
        }
        // otherwise, no change
        newSums[tid].feat[dimIndex] = 0.0;
    }
    newCounts[tid] = 0;
}

/*
this kernel is used to test performance differences between
the yinyang centroid update and the standard centroid update 
*/
__global__ void calcNewCentroidsAve(point *dataset,
                                    cent *centroidDataset,
                                    vector *newSums,
                                    unsigned int *newCounts,
                                    DTYPE *maxDriftArr)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);

    if(tid >= NCLUST)
    return;

    unsigned int dimIndex;

    DTYPE compDrift;

    struct vector oldVec; 

    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        if(newCounts[tid] > 0)
        {
            oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
        
            centroidDataset[tid].vec.feat[dimIndex] =
                newSums[tid].feat[dimIndex] / newCounts[tid];

            newSums[tid].feat[dimIndex] = 0.0;
        }
        else
        {
            oldVec.feat[dimIndex] = centroidDataset[tid].vec.feat[dimIndex];
        
            centroidDataset[tid].vec.feat[dimIndex] = 0.0;

            newSums[tid].feat[dimIndex] = 0.0;
        }

    }

    // compute drift
    compDrift = calcDis(&oldVec, &centroidDataset[tid].vec);
    centroidDataset[tid].drift = compDrift;

    atomicMax(&maxDriftArr[centroidDataset[tid].groupNum], compDrift);

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
__global__ void checkConverge(point * dataset,
                              unsigned int * conFlag)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    if(dataset[tid].oldCentroid != dataset[tid].centroidIndex)
    atomicCAS(conFlag, 0, 1);
    
}

/*
simple helper kernel that clears the drift array of size T
on the GPU. Called once each iteration for a total of MAXITER times
*/
__global__ void clearDriftArr(DTYPE *devMaxDriftArr)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NGROUP)
    return;
    
    devMaxDriftArr[tid] = 0.0;
}

__global__ void clearCentCalcData(vector *newCentSum,
                                  vector *oldCentSum,
                                  unsigned int *newCentCount,
                                  unsigned int *oldCentCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NCLUST)
    return;

    unsigned int dimIndex;

    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        newCentSum[tid].feat[dimIndex] = 0.0;
        oldCentSum[tid].feat[dimIndex] = 0.0;
    }
    newCentCount[tid] = 0;
    oldCentCount[tid] = 0;

}

__global__ void clearCentCalcDataLloyd(vector *newCentSum,
                                       unsigned int *newCentCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NCLUST)
    return;

    unsigned int dimIndex;
    
    for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
    {
        newCentSum[tid].feat[dimIndex] = 0.0;
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
__device__ DTYPE calcDis(vector *vec1, 
                         vector *vec2)
{
    unsigned int index;
    DTYPE total = 0;
    DTYPE square;

    for(index = 0; index < NDIM; index++)
    {
        square = (vec1->feat[index] - vec2->feat[index]);
        total += square * square;
    }

    return sqrt(total);
}



/////////////////////////////////////////////////////////////////////////
// Overloaded kernels and functions for counting distance calculations //
/////////////////////////////////////////////////////////////////////////

/*
global kernel that assigns data points to their first
centroid assignment. Runs exactly once.
*/
__global__ void initRunKernel(point * dataset,
                              cent * centroidDataset,
                              unsigned long long int *calcCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    unsigned int centIndex;

    DTYPE currDistance;
    dataset[tid].uprBound = INFINITY;

    for(centIndex = 0; centIndex < NCLUST; centIndex++)
    {
        // calculate euclidean distance between point and centroid
        currDistance = calcDis(&dataset[tid].vec, 
                               &centroidDataset[centIndex].vec);
        atomicAdd(calcCount, 1);
        if(currDistance < dataset[tid].uprBound)
        {
            // make the former current min the new 
            // lower bound for it's group
            if(dataset[tid].uprBound != INFINITY)
            dataset[tid].lwrBoundArr[centroidDataset[dataset[tid].centroidIndex].groupNum] = dataset[tid].uprBound;

            // update assignment and upper bound
            dataset[tid].centroidIndex = centIndex;
            dataset[tid].uprBound = currDistance;
            dataset[tid].uprBound = currDistance;
        }
        else if(currDistance < dataset[tid].lwrBoundArr[centroidDataset[centIndex].groupNum])
        {
            dataset[tid].lwrBoundArr[centroidDataset[centIndex].groupNum] = currDistance;
        }  
    }
}

// overload of assignPointsFull for counting distance calculations performed
__global__ void assignPointsFull(point *dataset,
                                 cent *centroidDataset,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    // variable declaration
    unsigned int btid = threadIdx.x;
    dataset[tid].oldCentroid = dataset[tid].centroidIndex;
    DTYPE tmpGlobLwr = INFINITY;
    unsigned int index;

    __shared__ unsigned int groupLclArr[NGROUP*BLOCKSIZE];

    // update points upper bound
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for(index = 0; index < NGROUP; index++)
    {        
        // subtract lowerbound by group's drift
        dataset[tid].lwrBoundArr[index] -= maxDriftArr[index];

        // if the lowerbound is less than the temp global lower,
        if(dataset[tid].lwrBoundArr[index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = dataset[tid].lwrBoundArr[index];
        }
    }

    // if the global lower bound is less than the upper bound
    if(tmpGlobLwr < dataset[tid].uprBound)
    {
        // tighten upper bound ub = d(x, b(x))
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                    &centroidDataset[dataset[tid].centroidIndex].vec);

        atomicAdd(calcCount, 1);

        // if the lower bound is less than the upper bound
        if(tmpGlobLwr < dataset[tid].uprBound)
        {
            // loop through groups
            for(index = 0; index < NGROUP; index++)
            {
                // if the lower bound is less than the upper bound
                if(dataset[tid].lwrBoundArr[index] < dataset[tid].uprBound)
                {
                    // that lower bounds group is marked
                    groupLclArr[index + (btid * NGROUP)] = 1;
                }
                else
                {
                    groupLclArr[index + (btid * NGROUP)] = 0;
                }
            }
            // execute point calcs given the groups
            pointCalcsFullAlt(&dataset[tid],
                           centroidDataset,
                           &groupLclArr[btid * NGROUP],
                           maxDriftArr,
                           calcCount);
        }
    }
}

// overload of assignPointsSimple for counting distance calculations performed
__global__ void assignPointsSimple(point *dataset,
                                   cent *centroidDataset,
                                   DTYPE *maxDriftArr,
                                   unsigned long long int *calcCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    DTYPE tmpGlobLwr = INFINITY;
    
    unsigned int btid = threadIdx.x;
    unsigned int index;

    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    __shared__ unsigned int groupLclArr[NGROUP*BLOCKSIZE];

    // update points upper bound
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;

    // update group lower bounds
    // for all in lower bound array
    for(index = 0; index < NGROUP; index++)
    {
        // subtract lowerbound by group's drift
        dataset[tid].lwrBoundArr[index] -= maxDriftArr[index];
        // if the lowerbound is less than the temp global lower,
        if(dataset[tid].lwrBoundArr[index] < tmpGlobLwr)
        {
            // lower bound is new temp global lower
            tmpGlobLwr = dataset[tid].lwrBoundArr[index];
        }
    }

    // if the global lower bound is less than the upper bound
    if(tmpGlobLwr < dataset[tid].uprBound)
    {

        // tighten upper bound ub = d(x, b(x))
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                &centroidDataset[dataset[tid].centroidIndex].vec);

        atomicAdd(calcCount, 1);


        // if the lower bound is less than the upper bound
        if(tmpGlobLwr < dataset[tid].uprBound)
        {
            // loop through groups
            for(index = 0; index < NGROUP; index++)
            {
                // if the lower bound is less than the upper bound
                if(dataset[tid].lwrBoundArr[index] < dataset[tid].uprBound)
                {
                    // that lower bounds group is marked
                    groupLclArr[index + (btid * NGROUP)] = 1;
                }
                else
                {
                    groupLclArr[index + (btid * NGROUP)] = 0;
                }
            }
            // execute point calcs given the groups
            pointCalcsSimple(&dataset[tid],
                             centroidDataset,
                             &groupLclArr[btid * NGROUP],
                             maxDriftArr,
                             calcCount);
        }
    }
}

// overload of assignPointsSimple for counting distance calculations performed
__global__ void assignPointsSuper(point *dataset,
                                  cent *centroidDataset,
                                  DTYPE *maxDrift,
                                  unsigned long long int *calcCount)
{
    unsigned int tid=threadIdx.x+(blockIdx.x*BLOCKSIZE);
    if(tid >= NPOINT)
    return;

    // set centroid's old centroid to be current assignment
    dataset[tid].oldCentroid = dataset[tid].centroidIndex;

    // point calc variables
    unsigned int clustIndex;
    DTYPE compDistance;

    // update bounds
    dataset[tid].uprBound += centroidDataset[dataset[tid].centroidIndex].drift;
    dataset[tid].lwrBoundArr[0] -= *maxDrift;


    if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
    {
        // tighten upper bound
        dataset[tid].uprBound =
            calcDis(&dataset[tid].vec,
                    &centroidDataset[dataset[tid].centroidIndex].vec);
        atomicAdd(calcCount, 1);


        if(dataset[tid].lwrBoundArr[0] < dataset[tid].uprBound)
        {
            // to get a new lower bound
            dataset[tid].lwrBoundArr[0] = INFINITY;

            for(clustIndex = 0; clustIndex < NCLUST; clustIndex++)
            {
                // do not calculate for the already assigned cluster
                if(clustIndex == dataset[tid].oldCentroid)
                continue;

                compDistance = calcDis(&dataset[tid].vec,
                                       &centroidDataset[clustIndex].vec);
                atomicAdd(calcCount, 1);
                
                if(compDistance < dataset[tid].uprBound)
                {
                    dataset[tid].lwrBoundArr[0] = dataset[tid].uprBound;
                    dataset[tid].centroidIndex = clustIndex;
                    dataset[tid].uprBound = compDistance;
                }
                else if(compDistance < dataset[tid].lwrBoundArr[0])
                {
                    dataset[tid].lwrBoundArr[0] = compDistance;
                }
            }
        }
    }
}

// overload of pointCalcsFull for counting distance calculations debug
__device__ void pointCalcsFull(point *pointPtr,
                               cent *centroidDataset,
                               unsigned int *groupArr,
                               DTYPE *maxDriftArr,
                               unsigned long long int *calcCount)
{

    unsigned int grpIndex, clstIndex;
    
    DTYPE compDistance;
    DTYPE oldLwr;
    DTYPE oldCentUpr = pointPtr->uprBound;
    DTYPE oldCentLwr = pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum];

    // loop through all the groups
    for(grpIndex = 0; grpIndex < NGROUP; grpIndex++)
    {
        // if the group is marked as going through the group filter
        if(groupArr[grpIndex])
        {
            // save the former lower bound pre-update
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum)
            oldLwr = oldCentLwr + maxDriftArr[grpIndex];
            else
            oldLwr = pointPtr->lwrBoundArr[grpIndex] + maxDriftArr[grpIndex];

            // reset the group's lower bound in order to find the new lower bound
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;

            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;

            // loop through all the group's clusters
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                // if the cluster is the cluster already assigned 
                // at the start of this iteration
                if(clstIndex == pointPtr->oldCentroid)
                continue;

                // if the cluster is a part of the group being checked now
                if(grpIndex == centroidDataset[clstIndex].groupNum)
                {
                    // local filtering condition
                    if(pointPtr->lwrBoundArr[grpIndex] < oldLwr - centroidDataset[clstIndex].drift)
                    continue;

                    // perform distance calculation
                    compDistance = calcDis(&pointPtr->vec, &centroidDataset[clstIndex].vec);
                    atomicAdd(calcCount, 1);

                    if(compDistance < pointPtr->uprBound)
                    {
                        pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                        pointPtr->centroidIndex = clstIndex;
                        pointPtr->uprBound = compDistance;
                    }
                    else if(compDistance < pointPtr->lwrBoundArr[grpIndex])
                    {
                        pointPtr->lwrBoundArr[grpIndex] = compDistance;
                    }
                }
            }
        }
        // this is a very rare condition in which an assigned centroid was in a group that did not pass through the group filter
        else
        {
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;
        }
    }
}

// overload for distance calculation counting debug
__device__ void pointCalcsSimple(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount)
{

    unsigned int index;
    DTYPE compDistance;

    for(index = 0; index < NGROUP; index++)
    {
        if(groupArr[index])
        {
            pointPtr->lwrBoundArr[index] = INFINITY;
        }
    }

    for(index = 0; index < NCLUST; index++)
    {
        if(groupArr[centroidDataset[index].groupNum])
        {
            if(index == pointPtr->oldCentroid)
            continue;

            compDistance = calcDis(&pointPtr->vec, 
                                   &centroidDataset[index].vec);
            atomicAdd(calcCount, 1);

            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = index;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[index].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[index].groupNum] = compDistance;
            }
        }
    }
}




// overload for distance calculation counting debug
__device__ void pointCalcsFullAlt(point *pointPtr,
                                 cent *centroidDataset,
                                 unsigned int *groupArr,
                                 DTYPE *maxDriftArr,
                                 unsigned long long int *calcCount)
{
    unsigned int index;
    DTYPE compDistance;
    DTYPE oldLwrs[NGROUP];

    for(index = 0; index < NGROUP; index++)
    {
        if(groupArr[index])
        {
            oldLwrs[index] = pointPtr->lwrBoundArr[index] + maxDriftArr[index];
            pointPtr->lwrBoundArr[index] = INFINITY;
        }
    }

    for(index = 0; index < NCLUST; index++)
    {
        if(groupArr[centroidDataset[index].groupNum])
        {
            if(index == pointPtr->oldCentroid)
            continue;

            if(pointPtr->lwrBoundArr[centroidDataset[index].groupNum] < oldLwrs[centroidDataset[index].groupNum] - centroidDataset[index].drift)
            continue;

            compDistance = calcDis(&pointPtr->vec,
                                   &centroidDataset[index].vec);
            atomicAdd(calcCount,1);

            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = index;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[index].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[index].groupNum] = compDistance;
            }
        }
    }
}
