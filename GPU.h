#ifndef GPU_H
#define GPU_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <unistd.h>

#include "params.h"
#include "kernel.h"
#include "kmeansUtils.h"
#include "omp.h"

void warmupGPU();

void storeDataOnGPU(point * dataset, 
                    unsigned long int numPnt);

point *storeStructDataOnGPU(point *dataset, 
                            unsigned long int numPnt);

cent *storeCentDataOnGPU(cent *centDataset,
                         const unsigned int numPnt);

int startSimpleOnGPU(point *hostDataset,
                     cent *hostCentDataset,
                     unsigned long long int *calcCount,
                     double *simpStartTime,
                     double *simpEndTime,
                     unsigned int *ranIter);

int startFullOnGPU(point *hostDataset,
                     cent *hostCentDataset,
                     unsigned long long int *calcCount,
                     double *simpStartTime,
                     double *simpEndTime,
                     unsigned int *ranIter);

int startLloydOnGPU(point *hostDataset,
                    cent *hostCentDataset,
                    double *lloydStartTime,
                    double *lloydEndTime,
                    unsigned int *ranIter);



int startSuperOnGPU(point *hostDataset,
                    cent *hostCentDataset,
                    unsigned long long int *calcCount,
                    double *supStartTime,
                    double *supEndTime,
                    unsigned int *ranIter);

#endif
