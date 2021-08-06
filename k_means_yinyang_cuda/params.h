#ifndef PARAMS_H
#define PARAMS_H

// precision of floating point data type
// double or float
#define DTYPE double

//Kernel block size
#define BLOCKSIZE 256

// RNG seed
#define SEED 73

// constants for algorithms
typedef enum
{
  FULLGPU = 100000,
  FULLCPU,
  SIMPLEGPU,
  SIMPLECPU,
  SUPERGPU,
  SUPERCPU,
  LLOYDGPU,
  LLOYDCPU,
  INVALIDIMP
} ImpType;


// Struct Data Types //

//Centroid
typedef struct CentInfo
{
	//Centroid's group index
	int groupNum;

	//Centroid's drift after updating
	DTYPE drift;

  //number of data points assigned to centroid
	int count;
} cent;

//Struct representing a datapoint
typedef struct PointInfo
{
	//Indices of old and new assigned centroids
	int centroidIndex;
	int oldCentroid;

	//The current upper bound
	DTYPE uprBound;
}point;


#endif
