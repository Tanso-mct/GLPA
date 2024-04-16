#ifndef DEVICE_FUNCS_CUH_
#define DEVICE_FUNCS_CUH_

#include "cg.h"
#include <cmath>
#include <Windows.h>

extern __device__ void mtProduct4x4Vec3d(
    float* mt4x4,
    float* vec,
    float* result
);

extern __device__ void vecGetVecsCos(
    float* vec1,
    float* vec2,
    float* result
);

extern __device__ int judgePolyVInViewVolume(
    float* cnvtPolyV,
    float camFarZ,
    float camNearZ,
    float* camViewAngle
);

#endif DEVICE_FUNCS_CUH_