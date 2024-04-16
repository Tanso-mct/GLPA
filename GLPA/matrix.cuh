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

// float ary mt4x4, float ary vec, float result
#define MT_PRODUCT_4X4_VEC3D(mt4x4, vec, result) \
    do { \
        result[0] = vec[AX] * mt4x4[0] + vec[AY] * mt4x4[1] + vec[AZ] * mt4x4[2] + 1 * mt4x4[3]; \
        result[1] = vec[AX] * mt4x4[4] + vec[AY] * mt4x4[5] + vec[AZ] * mt4x4[6] + 1 * mt4x4[7]; \
        result[2] = vec[AX] * mt4x4[8] + vec[AY] * mt4x4[9] + vec[AZ] * mt4x4[10] + 1 * mt4x4[11]; \
    } while(0);
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

