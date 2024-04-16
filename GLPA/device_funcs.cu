#include "device_funcs.cuh"

__device__ void mtProduct4x4Vec3d(
    float* mt4x4,
    float* vec,
    float* result
){
    result[0] = vec[AX] * mt4x4[0] + vec[AY] * mt4x4[1] + vec[AZ] * mt4x4[2] + 1 * mt4x4[3];
    result[1] = vec[AX] * mt4x4[4] + vec[AY] * mt4x4[5] + vec[AZ] * mt4x4[6] + 1 * mt4x4[7];
    result[2] = vec[AX] * mt4x4[8] + vec[AY] * mt4x4[9] + vec[AZ] * mt4x4[10] + 1 * mt4x4[11];
}

__device__ void vecGetVecsCos(
    float* vec1,
    float* vec2,
    float* result
){
    *result
    = (vec1[AX] * vec2[AX] + vec1[AY] * vec2[AY] + vec1[AZ] * vec2[AZ]) /
    (sqrt(vec1[AX] * vec1[AX] + vec1[AY] * vec1[AY] + vec1[AZ] * vec1[AZ]) * 
    sqrt(vec2[AX] * vec2[AX] + vec2[AY] * vec2[AY] + 
    vec2[AZ] * vec2[AZ]));
}

__device__ int judgePolyVInViewVolume(
    float* cnvtPolyV,
    float camFarZ,
    float camNearZ,
    float* camViewAngle
){
    float cnvtXzPolyV[3] = {cnvtPolyV[AX], 0, cnvtPolyV[AZ]};
    float cnvtYzPolyV[3] = {0, cnvtPolyV[AY], cnvtPolyV[AZ]};

    float zVec[3] = {0, 0, -1};

    float cnvtXzPolyVxZVecDotCos;
    vecGetVecsCos(zVec, cnvtXzPolyV, &cnvtXzPolyVxZVecDotCos);

    float cnvtYzPolyVxZVecDotCos;
    vecGetVecsCos(zVec, cnvtYzPolyV, &cnvtYzPolyVxZVecDotCos);

    int polyVZInIF = (cnvtPolyV[AZ] >= -camFarZ && cnvtPolyV[AZ] <= -camNearZ) ? TRUE : FALSE;
    int polyXzVInIF = (cnvtXzPolyVxZVecDotCos >= camViewAngle[AX]) ? TRUE : FALSE;
    int polyYzVInIF = (cnvtYzPolyVxZVecDotCos >= camViewAngle[AY]) ? TRUE : FALSE;

    return (polyVZInIF == TRUE && polyXzVInIF == TRUE && polyYzVInIF == TRUE) ? TRUE : FALSE;
}