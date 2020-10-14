#ifndef PARAMS_H
#define PARAMS_H

//Parameters

//Kernel block size
#define BLOCKSIZE 256

//Number of data dimensions
#define NDIM 8

//Number of datapoints
#define NPOINT 1000000

//Number of clusters
#define NCLUST 100

//Number of groups for gpu implementation
#define NGROUP 10

//Number of groups for cpu implementation
#define NGROUPCPU 100

//max limit of number of kmeans iterations
#define MAXITER 3000

// RNG seed
#define SEED 73

// number of cpu threads
#define NTHREAD 16

//Structs

//Vector with NDIM number of features
struct vector
{
	double feat[NDIM];
};

//Centroid
struct cent
{
	//Centroid's pointdata
	struct vector vec;

	//Centroid's group index
	int groupNum;

	//Centroid's drift after updating
	double drift;

	int count;
};


//Struct representing a datapoint
struct point
{
	//Point's feature vector
	struct vector vec;

	//Indices of old and new assigned centroids
	int centroidIndex;
	int oldCentroid;

	//Array of lower bound distances
	double lwrBoundArr[NGROUPCPU];

	//The current upper bound
	double uprBound;
};

#endif
