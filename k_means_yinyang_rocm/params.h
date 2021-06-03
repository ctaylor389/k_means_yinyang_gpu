#ifndef PARAMS_H
#define PARAMS_H

// select which NGROUP will be used (NGROUP or NGROUPCPU)
// this is for declaration of point datatype
#define GROUPTYPE NGROUP

// precision of floating point data type
// double or float
#define DTYPE double

//Parameters

//Kernel block size
#define BLOCKSIZE 256

//Number of data dimensions
#define NDIM 2

//Number of datapoints
#define NPOINT 1864620

//Number of clusters
#define NCLUST 200

//Number of groups (NCLUST / 10 for CPU) (20 for GPU) (1 for Super Simplified Implementations)
#define NGROUP 20

#define NGROUPCPU 20

//Number of kmeans iterations
#define MAXITER 3000

// RNG seed
#define SEED 73

// number of cpu threads
#define NTHREAD 16

// number of executions for averaging time
#define NRUNS 1


// Struct Data Types //

//Vector with NDIM number of features
typedef struct vector
{
	DTYPE feat[NDIM];
}vector;

//Centroid
typedef struct cent
{
	//Centroid's pointdata
	vector vec;

	//Centroid's group index
	int groupNum;

	//Centroid's drift after updating
	DTYPE drift;

	int count;
} cent;

//Struct representing a datapoint
typedef struct point
{
	//Point's feature vector
	vector vec;

	//Indices of old and new assigned centroids
	int centroidIndex;
	int oldCentroid;

	//Array of lower bound distances
	DTYPE lwrBoundArr[GROUPTYPE];

	//The current upper bound
	DTYPE uprBound;
}point;

#endif
