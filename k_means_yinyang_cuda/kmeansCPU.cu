#include "kmeansCPU.h"



/////////////////////////////////
// CPU k-means implementations //
/////////////////////////////////
int startLloydOnCPU(point *dataset,
                     cent *centroidDataset,
                     double *startTime,
                     double *endTime,
                     unsigned int *ranIter)
{

    *startTime = omp_get_wtime();
    
    unsigned int pntIndex, clstIndex, dimIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;


    omp_set_num_threads(NTHREAD);

    DTYPE currMin, currDis;

    // start standard kmeans algorithm for MAXITER iterations
    while(!conFlag && index < MAXITER)
    {
        currMin = INFINITY;

        #pragma omp parallel private(pntIndex, clstIndex, currDis, currMin) shared(dataset, centroidDataset)
        {
            #pragma omp for nowait schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;
                for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
                {
                    currDis = calcDisCPU(dataset[pntIndex].vec, 
                                         centroidDataset[clstIndex].vec,
                                         NDIM);
                    if(currDis < currMin)
                    {
                        dataset[pntIndex].centroidIndex = clstIndex;
                        currMin = currDis;
                    }
                }
                currMin = INFINITY;
            }
        }

        // clear centroids features
        for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
        {
            for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
            {
                centroidDataset[clstIndex].vec.feat[dimIndex] = 0.0;
            }
            centroidDataset[clstIndex].count = 0;
        }
        // sum all assigned point's features
        #pragma omp parallel private(pntIndex, dimIndex) shared(dataset, centroidDataset)
        {
            #pragma omp for nowait schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                #pragma omp atomic
                centroidDataset[dataset[pntIndex].centroidIndex].count++;

                for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
                {
                    #pragma omp atomic
                    centroidDataset[dataset[pntIndex].centroidIndex].vec.feat[dimIndex] +=
                        dataset[pntIndex].vec.feat[dimIndex];
                }
            }
        }
        // take the average of each feature to get new centroid features
        for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
        {
            for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
            {
                if(centroidDataset[clstIndex].count > 0)
                {
                    centroidDataset[clstIndex].vec.feat[dimIndex] /= 
                                        centroidDataset[clstIndex].count;
                }
                // otherwise, centroid remains the same
            }
        }
        index++;
        conFlag = checkConverge(dataset);
    }

    *endTime = omp_get_wtime();
    *ranIter = index;

    //printf("iterations: %d\n", index);
    return 0;
}

int startFullOnCPU(point *dataset,
                   cent *centroidDataset,
                   double *startTime,
                   double *endTime,
                   unsigned int *ranIter)
{
    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, grpIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;

    // array to contain the maximum drift of each group of centroids
    // note: shared amongst all points
    DTYPE driftArr[NGROUPCPU];

    // array to contain integer flags which mark which groups need to be checked
    // for a potential new centroid
    // note: unique to each point
    int groupLclArr[NGROUPCPU];

    omp_set_num_threads(NTHREAD);

    // the minimum of all the lower bounds for a single point
    DTYPE tmpGlobLwr = INFINITY;
    
    // cluster the centroids into NGROUPCPU groups
    groupCent(centroidDataset, NCLUST, NGROUPCPU, NDIM);

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // clear drift array each new iteration
        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
        {
            driftArr[grpIndex] = 0.0;
        }
    
        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, driftArr);

        // filtering done in parallel
        #pragma omp parallel private(pntIndex, grpIndex, tmpGlobLwr, groupLclArr) shared(dataset, centroidDataset, driftArr)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

                tmpGlobLwr = INFINITY;
                
                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                    centroidDataset[dataset[pntIndex].centroidIndex].drift;
                
                // update group lower bounds
                    // lb = lb - maxGroupDrift
                for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                {
                    dataset[pntIndex].lwrBoundArr[grpIndex] -= driftArr[grpIndex];

                    if(dataset[pntIndex].lwrBoundArr[grpIndex] < tmpGlobLwr)
                    {
                        // minimum lower bound
                        tmpGlobLwr = dataset[pntIndex].lwrBoundArr[grpIndex];
                    }
                }

                // global filtering
                // if global lowerbound >= upper bound
                if(tmpGlobLwr < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec,
                                   centroidDataset[dataset[pntIndex].centroidIndex].vec,
                                   NDIM);

                    // check condition again
                    if(tmpGlobLwr < dataset[pntIndex].uprBound)
                    {
                        // group filtering
                        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                        {
                            // mark groups that need to be checked
                            if(dataset[pntIndex].lwrBoundArr[grpIndex] < dataset[pntIndex].uprBound)
                            groupLclArr[grpIndex] = 1;
                            else
                            groupLclArr[grpIndex] = 0;
                        }

                        // pass group array and point to go execute distance calculations
                        pointCalcsFull(&dataset[pntIndex], groupLclArr, 
                                                driftArr, centroidDataset);
                    }
                }
            }
        }
        index++;
        conFlag = checkConverge(dataset);
    }
    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;
}



/*
CPU implementation of the simplified Yinyang algorithm
given only a file name with the points and a start and end time
*/
int startSimpleOnCPU(point *dataset,
                     cent *centroidDataset,
                     double *startTime,
                     double *endTime,
                     unsigned int *ranIter)
{
    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, grpIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;

    // array to contain the maximum drift of each group of centroids
    // note: shared amongst all points
    DTYPE driftArr[NGROUPCPU];

    // array to contain integer flags which mark which groups need to be checked
    // for a potential new centroid
    // note: unique to each point
    int groupLclArr[NGROUPCPU];

    omp_set_num_threads(NTHREAD);
    
    // the minimum of all the lower bounds for a single point
    DTYPE tmpGlobLwr = INFINITY;

    // cluster the centroids int NGROUP groups
    groupCent(centroidDataset, NCLUST, NGROUPCPU, NDIM);

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // clear drift array each new iteration
        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
        {
            driftArr[grpIndex] = 0.0;
        }

        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, driftArr);

        // filtering done in parallel
        #pragma omp parallel private(pntIndex, grpIndex, tmpGlobLwr, groupLclArr) shared(dataset, centroidDataset, driftArr)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;
                tmpGlobLwr = INFINITY;

                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                        centroidDataset[dataset[pntIndex].centroidIndex].drift;

                // update group lower bounds and find min lower bound
                    // lb = lb - maxGroupDrift
                for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                {
                    dataset[pntIndex].lwrBoundArr[grpIndex] -= driftArr[grpIndex];

                    if(dataset[pntIndex].lwrBoundArr[grpIndex] < tmpGlobLwr)
                    {
                        // minimum lower bound
                        tmpGlobLwr = dataset[pntIndex].lwrBoundArr[grpIndex];
                    }
                }

                // global filtering
                // if global lowerbound >= upper bound
                if(tmpGlobLwr < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec, 
                            centroidDataset[dataset[pntIndex].centroidIndex].vec,
                            NDIM);
                    
                    // check condition again
                    if(tmpGlobLwr < dataset[pntIndex].uprBound)
                    {
                        // group filtering
                        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                        {
                            // mark groups that need to be checked
                            if(dataset[pntIndex].lwrBoundArr[grpIndex] < dataset[pntIndex].uprBound)
                            groupLclArr[grpIndex] = 1;
                            else
                            groupLclArr[grpIndex] = 0;
                        }

                        // pass group array and point to go execute distance calculations
                        pointCalcsSimple(&dataset[pntIndex],
                                         groupLclArr,
                                         driftArr,
                                         centroidDataset);
                    }
                }
            }
        }
        index++;
        conFlag = checkConverge(dataset);
    }
    
    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;
}


int startSuperOnCPU(point *dataset,
                    cent *centroidDataset,
                    double *startTime,
                    double *endTime,
                    unsigned int *ranIter)
{
    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, clstIndex;
    unsigned int index = 0;

    DTYPE compDistance;
    DTYPE maxDrift = 0.0;

    unsigned int conFlag = 0;

    omp_set_num_threads(NTHREAD);

    // place all centroids in one "group"
    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        centroidDataset[clstIndex].groupNum = 0;
    }

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, &maxDrift);
        
        // filtering done in parallel
        #pragma omp parallel private(pntIndex, clstIndex, compDistance) shared(dataset, centroidDataset, maxDrift)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                        centroidDataset[dataset[pntIndex].centroidIndex].drift;

                // update lower bound
                dataset[pntIndex].lwrBoundArr[0] -= maxDrift;

                // global filtering
                // if global lowerbound >= upper bound
                if(dataset[pntIndex].lwrBoundArr[0] < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec, 
                            centroidDataset[dataset[pntIndex].centroidIndex].vec,
                            NDIM);

                    // check condition again
                    if(dataset[pntIndex].lwrBoundArr[0] < dataset[pntIndex].uprBound)
                    {
                        dataset[pntIndex].lwrBoundArr[0] = INFINITY;

                        // calculate distance between point and every cluster
                        for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
                        {
                            // if clstIndex is the one already assigned, skip the calculation
                            if(clstIndex == dataset[pntIndex].oldCentroid)
                            continue;

                            compDistance = calcDisCPU(dataset[pntIndex].vec,
                                                      centroidDataset[clstIndex].vec,
                                                      NDIM);

                            if(compDistance < dataset[pntIndex].uprBound)
                            {
                                dataset[pntIndex].lwrBoundArr[0] = dataset[pntIndex].uprBound;
                                dataset[pntIndex].centroidIndex = clstIndex;
                                dataset[pntIndex].uprBound = compDistance;
                            }
                            else if(compDistance < dataset[pntIndex].lwrBoundArr[0])
                            {
                                dataset[pntIndex].lwrBoundArr[0] = compDistance;
                            }
                        }
                    }
                }			
            }
        }
        index++;
        conFlag = checkConverge(dataset);
    }

    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;


}


unsigned int checkConverge(point *dataset)
{
    int index;
    {
        for(index = 0; index < NPOINT; index++)
        {
            if(dataset[index].centroidIndex != dataset[index].oldCentroid)
            {
                return 0;
            }
        }
    }
    return 1;
}



void pointCalcsFull(point *pointPtr,
                    int *groupArr,
                    DTYPE *driftArr,
                    cent *centroidDataset)
{
    // index variables
    unsigned int clstIndex, grpIndex;

    DTYPE compDistance;
    DTYPE oldLwr;
    DTYPE oldCentUpr = pointPtr->uprBound;
    DTYPE oldCentLwr = pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum];

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {        
        if(groupArr[grpIndex])
        {
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum)
            oldLwr = oldCentLwr + driftArr[grpIndex];
            else
            oldLwr = pointPtr->lwrBoundArr[grpIndex] + driftArr[grpIndex];
                        
            // set group's lower bound to find new lower bound for this iteration
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;

            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;

            // loop through all of the group's centroids
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                if(clstIndex == pointPtr->oldCentroid)
                continue;

                if(centroidDataset[clstIndex].groupNum == grpIndex)
                {
                    // local filtering condition
                    if(pointPtr->lwrBoundArr[grpIndex] < oldLwr  - centroidDataset[clstIndex].drift)
                    continue;

                    // perform distance calculation
                    compDistance = calcDisCPU(pointPtr->vec, centroidDataset[clstIndex].vec, NDIM);

                    if(compDistance < pointPtr->uprBound)
                    {
                        pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                        pointPtr->uprBound = compDistance;
                        pointPtr->centroidIndex = clstIndex;
                    }
                    else if(compDistance < pointPtr->lwrBoundArr[grpIndex])
                    {
                        pointPtr->lwrBoundArr[grpIndex] = compDistance;
                    }
                }
            }
        }
        else
        {
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;
        }
    }
}


/*
Uses more space but less branching
*/
void pointCalcsFullAlt(point *pointPtr,
                       int *groupArr,
                       DTYPE *driftArr,
                       cent *centroidDataset)
{
    // index variables
    unsigned int clstIndex, grpIndex;

    DTYPE compDistance;
    DTYPE oldLwrs[NGROUPCPU];

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        // if the group is not blocked by group filter
        if(groupArr[grpIndex])
        {
            oldLwrs[grpIndex] = pointPtr->lwrBoundArr[grpIndex] + driftArr[grpIndex];

            // reset the lwrBoundArr to be only new lwrBounds
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;
        }
    }

    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        // if the centroid's group is marked in groupArr
        if(groupArr[centroidDataset[clstIndex].groupNum])
        {
            // if it was the originally assigned cluster, no need to calc dist
            if(clstIndex == pointPtr->oldCentroid)
            continue;

            if(pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] < oldLwrs[centroidDataset[clstIndex].groupNum] - centroidDataset[clstIndex].drift)
            continue;
            
            // compute distance between point and centroid
            compDistance = calcDisCPU(pointPtr->vec, 
                                      centroidDataset[clstIndex].vec,
                                      NDIM);
            
            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = clstIndex;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum])
            {   
                pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] = compDistance;
            }
        }
    }    
}


void pointCalcsSimple(point *pointPtr,
                      int *groupArr,
                      DTYPE *driftArr,
                      cent *centroidDataset)
{
    // index variables
    unsigned int clstIndex, grpIndex;

    DTYPE compDistance;

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        // if the group is not blocked by group filter
        if(groupArr[grpIndex])
        {
            // reset the lwrBoundArr to be only new lwrBounds
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;
        }
    }
    
    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        // if the centroid's group is marked in groupArr
        if(groupArr[centroidDataset[clstIndex].groupNum])
        {
            // if it was the originally assigned cluster, no need to calc dist
            if(clstIndex == pointPtr->oldCentroid)
            continue;
        
            // compute distance between point and centroid
            compDistance = calcDisCPU(pointPtr->vec, 
                                      centroidDataset[clstIndex].vec,
                                      NDIM);

            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = clstIndex;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] = compDistance;
            }
        }
    }
}



/*
Function used to do an intial iteration of K-means
*/
void initPoints(point *dataset, 
                cent *centroidDataset)
{
    int pntIndex, clstIndex;

    DTYPE currDistance;
    
    // start single standard k-means iteration for initial bounds and cluster assignments
        // assignment
    #pragma omp parallel private(pntIndex, clstIndex, currDistance) shared(dataset, centroidDataset)
    {
        #pragma omp for schedule(static)
        for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
        {
            dataset[pntIndex].uprBound = INFINITY;

            // for all centroids
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                // currDistance is equal to the distance between the current feature
                // vector being inspected, and the current centroid being compared
                currDistance = calcDisCPU(dataset[pntIndex].vec, 
                                          centroidDataset[clstIndex].vec,
                                          NDIM);

                // if the the currDistance is less than the current minimum distance
                if(currDistance < dataset[pntIndex].uprBound)
                {
                    if(dataset[pntIndex].uprBound != INFINITY)
                    dataset[pntIndex].lwrBoundArr[centroidDataset[dataset[pntIndex].centroidIndex].groupNum] = dataset[pntIndex].uprBound;
                    // update assignment and upper bound
                    dataset[pntIndex].centroidIndex = clstIndex;
                    dataset[pntIndex].uprBound = currDistance;
                }
                else if(currDistance < dataset[pntIndex].lwrBoundArr[centroidDataset[clstIndex].groupNum])
                {
                    dataset[pntIndex].lwrBoundArr[centroidDataset[clstIndex].groupNum] = currDistance;
                }
            }
        }
    }
}

void updateCentroids(point *dataset, 
                     struct cent *centroidDataset, 
                     DTYPE *maxDriftArr)
{
    // holds the number of points assigned to each centroid formerly and currently
    int oldCounts[NCLUST];
    int newCounts[NCLUST];


    // comparison variables
    DTYPE compDrift;

    // holds the new vector calculated
    vector oldVec;
    DTYPE oldFeature;

    omp_set_num_threads(NTHREAD);

    omp_lock_t driftLock;

    omp_init_lock(&driftLock);
    

    // allocate data for new and old vector sums
    vector *oldSumPtr = 
            (struct vector *)malloc(sizeof(struct vector)*NCLUST);
    vector *newSumPtr = 
            (struct vector *)malloc(sizeof(struct vector)*NCLUST);


    DTYPE oldSumFeat;
    DTYPE newSumFeat;

    unsigned int pntIndex, clstIndex, grpIndex, dimIndex;
    
    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        maxDriftArr[grpIndex] = 0.0;
    }
    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
        {
            oldSumPtr[clstIndex].feat[dimIndex] = 0.0;
            newSumPtr[clstIndex].feat[dimIndex] = 0.0;
        }
    
        oldCounts[clstIndex] = 0;
        newCounts[clstIndex] = 0;
    }
    
    for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
    {
        // add one to the old count and new count for each centroid
        if(dataset[pntIndex].oldCentroid >= 0)
        oldCounts[dataset[pntIndex].oldCentroid]++;
        
        newCounts[dataset[pntIndex].centroidIndex]++;

        // if the old centroid does not match the new centroid,
        // add the points vector to each 
        if(dataset[pntIndex].oldCentroid != dataset[pntIndex].centroidIndex)
        {
            for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
            {
                if(dataset[pntIndex].oldCentroid >= 0)
                oldSumPtr[dataset[pntIndex].oldCentroid].feat[dimIndex] += 
                                    dataset[pntIndex].vec.feat[dimIndex];

                newSumPtr[dataset[pntIndex].centroidIndex].feat[dimIndex] += 
                                    dataset[pntIndex].vec.feat[dimIndex];
            }
        }
    }

    #pragma omp parallel private(clstIndex,dimIndex,oldVec,oldFeature,oldSumFeat,newSumFeat, compDrift) shared(driftLock,centroidDataset, maxDriftArr)
    {
    // create new centroid points
        #pragma omp for schedule(static)
        for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
        {
            for(dimIndex = 0; dimIndex < NDIM; dimIndex++)
            {
                if(newCounts[clstIndex] > 0)
                {
                    oldVec.feat[dimIndex] = centroidDataset[clstIndex].vec.feat[dimIndex];
                    oldFeature = oldVec.feat[dimIndex];
                    oldSumFeat = oldSumPtr[clstIndex].feat[dimIndex];
                    newSumFeat = newSumPtr[clstIndex].feat[dimIndex];

                    centroidDataset[clstIndex].vec.feat[dimIndex] = 
                    (oldFeature * oldCounts[clstIndex] - oldSumFeat + newSumFeat) 
                                                            / newCounts[clstIndex];
                }
                else
                {
                    // if the centroid has no current members, no change occurs to its position
                    oldVec.feat[dimIndex] = centroidDataset[clstIndex].vec.feat[dimIndex];
                }
            }

            compDrift = calcDisCPU(oldVec, centroidDataset[clstIndex].vec, NDIM);
            omp_set_lock(&driftLock);
            if(compDrift > maxDriftArr[centroidDataset[clstIndex].groupNum])
            {
                maxDriftArr[centroidDataset[clstIndex].groupNum] = compDrift;
            }
            omp_unset_lock(&driftLock);
            centroidDataset[clstIndex].drift = compDrift;
        }
    }
    omp_destroy_lock(&driftLock);

    free(newSumPtr);
    free(oldSumPtr);
}



//////////////////////////////////////////////////
// Overloads for counting distance calculations //
//////////////////////////////////////////////////

int startFullOnCPU(point *dataset,
                   cent *centroidDataset,
                   unsigned long long int *distCalcCount,
                   double *startTime,
                   double *endTime,
                   unsigned int *ranIter)
{
    // error bad input
    if(distCalcCount == NULL)
    return 1;

    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, grpIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;

    unsigned long long int pointCalcs;

    // array to contain the maximum drift of each group of centroids
    // note: shared amongst all points
    DTYPE driftArr[NGROUPCPU];

    // array to contain integer flags which mark which groups need to be checked
    // for a potential new centroid
    // note: unique to each point
    int groupLclArr[NGROUPCPU];

    omp_set_num_threads(NTHREAD);

    // the minimum of all the lower bounds for a single point
    DTYPE tmpGlobLwr = INFINITY;
    
    // cluster the centroids into NGROUPCPU groups
    groupCent(centroidDataset, NCLUST, NGROUPCPU, NDIM);

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset, distCalcCount);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // clear drift array each new iteration
        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
        {
            driftArr[grpIndex] = 0.0;
        }
    
        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, driftArr);

        // filtering done in parallel
        #pragma omp parallel private(pntIndex, grpIndex, tmpGlobLwr, groupLclArr, pointCalcs) shared(dataset, centroidDataset, driftArr)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

                tmpGlobLwr = INFINITY;

                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                        centroidDataset[dataset[pntIndex].centroidIndex].drift;

                // update group lower bounds
                    // lb = lb - maxGroupDrift
                for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                {
                    dataset[pntIndex].lwrBoundArr[grpIndex] -= driftArr[grpIndex];

                    if(dataset[pntIndex].lwrBoundArr[grpIndex] < tmpGlobLwr)
                    {
                        // assign temp global lowerBound
                        // for all lowerbounds assigned to point
                        tmpGlobLwr = dataset[pntIndex].lwrBoundArr[grpIndex];
                    }
                }

                // global filtering
                // if global lowerbound >= upper bound
                if(tmpGlobLwr < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec,
                                   centroidDataset[dataset[pntIndex].centroidIndex].vec,
                                   NDIM);

                    #pragma omp atomic
                    *distCalcCount+=1;

                    // check condition again
                    if(tmpGlobLwr < dataset[pntIndex].uprBound)
                    {
                        // group filtering
                        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                        {
                            // mark groups that need to be checked
                            if(dataset[pntIndex].lwrBoundArr[grpIndex] < dataset[pntIndex].uprBound)
                            groupLclArr[grpIndex] = 1;
                            else
                            groupLclArr[grpIndex] = 0;
                        }

                        // pass group array and point to go execute distance calculations
                        pointCalcs = pointCalcsFullCount(&dataset[pntIndex],
                                                         groupLclArr,
                                                         driftArr,
                                                         centroidDataset);

                        #pragma omp atomic
                        *distCalcCount+=pointCalcs;
                    }
                }
            }
        }
        index++;
        conFlag = checkConverge(dataset);
    }

    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;
}



/*
CPU implementation of the simplified Yinyang algorithm
given only a file name with the points and a start and end time
*/
int startSimpleOnCPU(point *dataset,
                     cent *centroidDataset,
                     unsigned long long int *distCalcCount,
                     double *startTime,
                     double *endTime,
                     unsigned int *ranIter)
{
    if(distCalcCount == NULL)
    return 1;

    unsigned long long int pointCalcs;

    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, grpIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;

    // array to contain the maximum drift of each group of centroids
    // note: shared amongst all points
    DTYPE driftArr[NGROUPCPU];

    // array to contain integer flags which mark which groups need to be checked
    // for a potential new centroid
    // note: unique to each point
    int groupLclArr[NGROUPCPU];

    omp_set_num_threads(NTHREAD);

    // the minimum of all the lower bounds for a single point
    DTYPE tmpGlobLwr = INFINITY;
    
    // cluster the centroids int NGROUP groups
    groupCent(centroidDataset, NCLUST, NGROUPCPU, NDIM);

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset, distCalcCount);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // clear drift array each new iteration
        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
        {
            driftArr[grpIndex] = 0.0;
        }
    
        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, driftArr);

        // filtering done in parallel
        #pragma omp parallel private(pntIndex, grpIndex, tmpGlobLwr, groupLclArr, pointCalcs) shared(dataset, centroidDataset, driftArr)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

                tmpGlobLwr = INFINITY;

                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                        centroidDataset[dataset[pntIndex].centroidIndex].drift;

                // update group lower bounds
                    // lb = lb - maxGroupDrift
                for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                {
                    dataset[pntIndex].lwrBoundArr[grpIndex] -= driftArr[grpIndex];

                    if(dataset[pntIndex].lwrBoundArr[grpIndex] < tmpGlobLwr)
                    {
                        // assign temp global lowerBound
                        // for all lowerbounds assigned to point
                        tmpGlobLwr = dataset[pntIndex].lwrBoundArr[grpIndex];
                    }
                }

                // global filtering
                // if global lowerbound >= upper bound
                if(tmpGlobLwr < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec, 
                            centroidDataset[dataset[pntIndex].centroidIndex].vec,
                            NDIM);

                    #pragma omp atomic
                    *distCalcCount+=1;

                    // check condition again
                    if(tmpGlobLwr < dataset[pntIndex].uprBound)
                    {
                        // group filtering
                        for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
                        {
                            // mark groups that need to be checked
                            if(dataset[pntIndex].lwrBoundArr[grpIndex] < dataset[pntIndex].uprBound)
                            groupLclArr[grpIndex] = 1;
                            else
                            groupLclArr[grpIndex] = 0;
                        }
                        // pass group array and point to go execute distance calculations
                        pointCalcs = pointCalcsSimpleCount(&dataset[pntIndex],
                                                           groupLclArr,
                                                           driftArr,
                                                           centroidDataset);
                        #pragma omp atomic
                        *distCalcCount+=pointCalcs;
                    }
                }
            }
        }
        index++;
        conFlag = checkConverge(dataset);
        printf("index %d: %llu calcs\n", index, *distCalcCount);
    }
    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;
}


int startSuperOnCPU(point *dataset,
                    cent *centroidDataset,
                    unsigned long long int *distCalcCount,
                    double *startTime,
                    double *endTime,
                    unsigned int *ranIter)
{

    if(distCalcCount == NULL)
    return 1;

    *startTime = omp_get_wtime();

    // index variables
    unsigned int pntIndex, clstIndex;
    unsigned int index = 0;
    unsigned int conFlag = 0;

    DTYPE compDistance;
    DTYPE maxDrift;

    omp_set_num_threads(NTHREAD);

    // place all centroids in one "group"
    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        centroidDataset[clstIndex].groupNum = 0;
    }

    // run one iteration of standard kmeans for initial centroid assignments
    initPoints(dataset, centroidDataset, distCalcCount);

    // master loop
    while(!conFlag && index < MAXITER)
    {
        // clear maxdrift
        maxDrift = 0.0;

        // update centers via optimised update method
        updateCentroids(dataset, centroidDataset, &maxDrift);

        // filtering done in parallel
        #pragma omp parallel private(pntIndex, clstIndex) shared(dataset, centroidDataset, maxDrift)
        {
            #pragma omp for schedule(static)
            for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
            {
                // reset old centroid before possibly finding a new one
                dataset[pntIndex].oldCentroid = dataset[pntIndex].centroidIndex;

                // update upper bound
                    // ub = ub + centroid's drift
                dataset[pntIndex].uprBound +=
                        centroidDataset[dataset[pntIndex].centroidIndex].drift;

                // update lower bound
                dataset[pntIndex].lwrBoundArr[0] -= maxDrift;

                // global filtering
                // if global lowerbound >= upper bound
                if(dataset[pntIndex].lwrBoundArr[0] < dataset[pntIndex].uprBound)
                {
                    // tighten upperbound ub = d(x, b(x))
                    dataset[pntIndex].uprBound = 
                        calcDisCPU(dataset[pntIndex].vec, 
                            centroidDataset[dataset[pntIndex].centroidIndex].vec,
                            NDIM);

                    #pragma omp atomic
                    *distCalcCount+=1;

                    // check condition again
                    if(dataset[pntIndex].lwrBoundArr[0] < dataset[pntIndex].uprBound)
                    {
                        dataset[pntIndex].lwrBoundArr[0] = INFINITY;

                        // calculate distance between point and every cluster
                        for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
                        {
                            // if clstIndex is the one already assigned, skip the calculation
                            if(clstIndex == dataset[pntIndex].oldCentroid)
                            continue;

                            compDistance = calcDisCPU(dataset[pntIndex].vec,
                                                      centroidDataset[clstIndex].vec,
                                                      NDIM);

                            #pragma omp critical
                            *distCalcCount+=1;

                            if(compDistance < dataset[pntIndex].uprBound)
                            {
                                dataset[pntIndex].lwrBoundArr[0] = dataset[pntIndex].uprBound;
                                dataset[pntIndex].centroidIndex = clstIndex;
                                dataset[pntIndex].uprBound = compDistance;
                            }
                            else if(compDistance < dataset[pntIndex].lwrBoundArr[0])
                            {
                                dataset[pntIndex].lwrBoundArr[0] = compDistance;
                            }
                        }
                    }
                }			
            }
        }
        index++;
        printf("index %d: %llu calcs\n", index, *distCalcCount);
        conFlag = checkConverge(dataset);
    }

    *endTime = omp_get_wtime();
    *ranIter = index + 1;

    return 0;


}

/*
Function used to do an intial iteration of K-means
*/
void initPoints(point *dataset, 
                cent *centroidDataset,
                unsigned long long int *distCalcCount)
{
    int pntIndex, clstIndex;

    DTYPE currDistance;
    
    // start single standard k-means iteration for initial bounds and cluster assignments
        // assignment
    #pragma omp parallel private(pntIndex, clstIndex, currDistance) shared(dataset, centroidDataset)
    {
        #pragma omp for schedule(static)
        for(pntIndex = 0; pntIndex < NPOINT; pntIndex++)
        {
            dataset[pntIndex].uprBound = INFINITY;

            // for all centroids
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                // currDistance is equal to the distance between the current feature
                // vector being inspected, and the current centroid being compared
                currDistance = calcDisCPU(dataset[pntIndex].vec, 
                                          centroidDataset[clstIndex].vec,
                                          NDIM);
		        #pragma omp atomic
		        *distCalcCount+=1;

                // if the the currDistance is less than the current minimum distance
                if(currDistance < dataset[pntIndex].uprBound)
                {
                    if(dataset[pntIndex].uprBound != INFINITY)
                    dataset[pntIndex].lwrBoundArr[centroidDataset[dataset[pntIndex].centroidIndex].groupNum] = dataset[pntIndex].uprBound;
                    // update assignment and upper bound
                    dataset[pntIndex].centroidIndex = clstIndex;
                    dataset[pntIndex].uprBound = currDistance;
                }
                else if(currDistance < dataset[pntIndex].lwrBoundArr[centroidDataset[clstIndex].groupNum])
                {
                    dataset[pntIndex].lwrBoundArr[centroidDataset[clstIndex].groupNum] = currDistance;
                }
            }
        }
    }
}


unsigned long long int pointCalcsFullCount(point *pointPtr,
                                           int *groupArr,
                                           DTYPE *driftArr,
                                           cent *centroidDataset)
{
    // index variables
    unsigned int clstIndex, grpIndex;
    unsigned long long int count = 0;

    DTYPE compDistance;
    DTYPE oldLwr;
    DTYPE oldCentUpr = pointPtr->uprBound;
    DTYPE oldCentLwr = pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum];

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        
        if(groupArr[grpIndex])
        {
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum)
            oldLwr = oldCentLwr + driftArr[grpIndex];
            else
            oldLwr = pointPtr->lwrBoundArr[grpIndex] + driftArr[grpIndex];
                        
            // set group's lower bound to find new lower bound for this iteration
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;

            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;

            // loop through all of the group's centroids
            for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
            {
                if(clstIndex == pointPtr->oldCentroid)
                continue;

                if(centroidDataset[clstIndex].groupNum == grpIndex)
                {
                    // local filtering condition
                    if(pointPtr->lwrBoundArr[grpIndex] < oldLwr  - centroidDataset[clstIndex].drift)
                    continue;

                    // perform distance calculation
                    compDistance = calcDisCPU(pointPtr->vec,
                                              centroidDataset[clstIndex].vec,
                                              NDIM);
                    count++;

                    if(compDistance < pointPtr->uprBound)
                    {
                        pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                        pointPtr->uprBound = compDistance;
                        pointPtr->centroidIndex = clstIndex;
                    }
                    else if(compDistance < pointPtr->lwrBoundArr[grpIndex])
                    {
                        pointPtr->lwrBoundArr[grpIndex] = compDistance;
                    }
                }
            }
        }
        else
        {
            if(grpIndex == centroidDataset[pointPtr->oldCentroid].groupNum && pointPtr->oldCentroid != pointPtr->centroidIndex)
            pointPtr->lwrBoundArr[centroidDataset[pointPtr->oldCentroid].groupNum] = oldCentUpr;
        }
    }
    return count;
}


/*
Uses more space but less branching
*/
unsigned long long int pointCalcsFullAltCount(point *pointPtr,
                                              int *groupArr,
                                              DTYPE *driftArr,
                                              cent *centroidDataset)
{
    unsigned long long int count = 0;
    unsigned int clstIndex, grpIndex;

    DTYPE compDistance;
    DTYPE oldLwrs[NGROUPCPU];

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        // if the group is not blocked by group filter
        if(groupArr[grpIndex])
        {
            oldLwrs[grpIndex] = pointPtr->lwrBoundArr[grpIndex] + driftArr[grpIndex];

            // reset the lwrBoundArr to be only new lwrBounds
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;
        }
    }

    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        // if the centroid's group is marked in groupArr
        if(groupArr[centroidDataset[clstIndex].groupNum])
        {
            // if it was the originally assigned cluster, no need to calc dist
            if(clstIndex == pointPtr->oldCentroid)
            continue;

            if(pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] < oldLwrs[centroidDataset[clstIndex].groupNum] - centroidDataset[clstIndex].drift)
            continue;
            
            // compute distance between point and centroid
            compDistance = calcDisCPU(pointPtr->vec, 
                                      centroidDataset[clstIndex].vec,
                                      NDIM);
            count++;
            
            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = clstIndex;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum])
            {   
                pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] = compDistance;
            }
        }

    }

    return count;    
}


unsigned long long int pointCalcsSimpleCount(point *pointPtr,
                                             int *groupArr,
                                             DTYPE *driftArr,
                                             cent *centroidDataset)
{
    unsigned long long int count = 0;
    unsigned int clstIndex, grpIndex;

    DTYPE compDistance;

    for(grpIndex = 0; grpIndex < NGROUPCPU; grpIndex++)
    {
        // if the group is not blocked by group filter
        if(groupArr[grpIndex])
        {
            // reset the lwrBoundArr to be only new lwrBounds
            pointPtr->lwrBoundArr[grpIndex] = INFINITY;
        }
    }

    for(clstIndex = 0; clstIndex < NCLUST; clstIndex++)
    {
        // if the centroid's group is marked in groupArr
        if(groupArr[centroidDataset[clstIndex].groupNum])
        {
            // if it was the originally assigned cluster, no need to calc dist
            if(clstIndex == pointPtr->oldCentroid)
            continue;
        
            // compute distance between point and centroid
            compDistance = calcDisCPU(pointPtr->vec, 
                                      centroidDataset[clstIndex].vec,
                                      NDIM);
            count++;
            if(compDistance < pointPtr->uprBound)
            {
                pointPtr->lwrBoundArr[centroidDataset[pointPtr->centroidIndex].groupNum] = pointPtr->uprBound;
                pointPtr->centroidIndex = clstIndex;
                pointPtr->uprBound = compDistance;
            }
            else if(compDistance < pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum])
            {
                pointPtr->lwrBoundArr[centroidDataset[clstIndex].groupNum] = compDistance;
            }
        }

    }
    return count; 
}
