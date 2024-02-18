#ifndef VECTOR_H_
#define VECTOR_H_

#include <vector>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cg.h"

__global__ void glpaGpuGetVecsCos(
    double* left_vec,
    double* right_vecs,
    double* result_vecs,
    int right_vecs_size
);

class Vector{
public :
    // host memory
    double* hLeftVec;
    double* hRightVec;
    double* hResult;

    // device memory
    double* dLeftVec;
    double* dRightVec;
    double* dResult;

    std::vector<double> getVecsCos(Vec3d left_vec, std::vector<Vec3d> right_vecs);
};

#endif  VECTOR_H_


