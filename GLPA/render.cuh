#ifndef RENDER_CUH_
#define RENDER_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "object.h"
#include "camera.cuh"

#include "cg.h"
#include "error.h"

// #include "device_funcs.cuh"

#include "vector.cuh"
#include "matrix.cuh"


__global__ void glpaGpuPrepareObj(
    int object_size,
    float* object_world_vs,
    float* matrix_camera_transformation_and_rotation,
    float camera_near_z,
    float camera_far_z,
    float* camera_view_angle_cos,
    int* result_object_in_view_volume_judge_ary
);

#define JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV, camFarZ, camNearZ, camViewAngle, polyVInIF) \
    do { \
        float cnvtXzPolyV[3] = {cnvtPolyV[AX], 0, cnvtPolyV[AZ]}; \
        float cnvtYzPolyV[3] = {0, cnvtPolyV[AY], cnvtPolyV[AZ]}; \
        \
        float zVec[3] = {0, 0, -1}; \
        \
        float cnvtXzPolyVxZVecDotCos; \
        VEC_GET_VECS_COS(zVec, cnvtXzPolyV, cnvtXzPolyVxZVecDotCos); \
        \
        float cnvtYzPolyVxZVecDotCos; \
        VEC_GET_VECS_COS(zVec, cnvtYzPolyV, cnvtYzPolyVxZVecDotCos); \
        \
        int polyVZInIF = (cnvtPolyV[AZ] >= -camFarZ && cnvtPolyV[AZ] <= -camNearZ) ? TRUE : FALSE; \
        int polyXzVInIF = (cnvtXzPolyVxZVecDotCos >= camViewAngle[AX]) ? TRUE : FALSE; \
        int polyYzVInIF = (cnvtYzPolyVxZVecDotCos >= camViewAngle[AY]) ? TRUE : FALSE; \
        \
        polyVInIF = (polyVZInIF == TRUE && polyXzVInIF == TRUE && polyYzVInIF == TRUE) ? TRUE : FALSE; \
    } while(0);


__global__ void glpaGpuRender(
    float* poly_vertices,
    float* poly_normals,
    int poly_amount,
    float* matrix_camera_transformation_and_rotation,
    float* matrix_camera_rotation,
    float camera_far_z,
    float camera_near_z,
    float* camera_view_angle,
    float* view_volume_vertices,
    float* view_volume_normals
);


class Render{
public :
    void prepareObjs(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam
    );

    void render(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam,
        LPDWORD buffer
    );

private :
    float* hMtCamTransRot;
    float* hMtCamRot;

    float* hCamViewAngleCos;
    int* hObjInJudgeAry;

};


#endif RENDER_CUH_