#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cg.h"


__global__ void glpaGpu4x4_4x1sMtProduct(
    double* mt_4x4,
    double* mt_4x1s,
    double* result_mt,
    int mt_4x1sSize
);


class Matrix{
public :
    // host memory
    double* hLeftMt;
    double* hRightMt;
    double* hResultMt;

    // device memory
    double* dLeftMt;
    double* dRightMt;
    double* dResultMt;

    std::vector<Vec3d> transRotConvert(Vec3d trans_vec, Vec3d rot_angle, std::vector<Vec3d> source_vecs);
    std::vector<Vec3d> rotConvert(Vec3d rot_angle, std::vector<Vec3d> source_vecs);
};


#endif MATRIX_H_

