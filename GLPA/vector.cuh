#ifndef VECTOR_H_
#define VECTOR_H_

#include <vector>
#include <math.h>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cg.h"
#include "error.h"

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

// float ary vec1, float ary vec2, float result
#define VEC_GET_VECS_COS(vec1, vec2, result) \
    do { \
        result \
        = (vec1[AX] * vec2[AX] + vec1[AY] * vec2[AY] + vec1[AZ] * vec2[AZ]) / \
        (sqrt(vec1[AX] * vec1[AX] + vec1[AY] * vec1[AY] + vec1[AZ] * vec1[AZ]) * \
        sqrt(vec2[AX] * vec2[AX] + vec2[AY] * vec2[AY] + \
        vec2[AZ] * vec2[AZ])); \
    } while(0);

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

    static bool ascending(double a, double b);
    static bool descending(double a, double b);


    std::vector<int> sortDecenOrder(std::vector<double>& source_nums);
    std::vector<int> sortAsenOrder(std::vector<double>& source_nums);
};

#endif  VECTOR_H_


