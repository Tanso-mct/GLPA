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

__global__ void glpaGpuGetSameSizeVecsCos(
    double* left_vecs,
    double* right_vecs,
    double* result_vecs,
    int vecs_size
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

    std::vector<double> getVecsDotCos(Vec3d left_vec, std::vector<Vec3d> right_vecs);
    std::vector<double> getSameSizeVecsDotCos(std::vector<Vec3d> left_vecs, std::vector<Vec3d> right_vecs);

    void pushVecToDouble(
        std::vector<Vec3d> source_vec,
        std::vector<double>* target_vec,
        int vec_i
    );
};

#endif  VECTOR_H_


