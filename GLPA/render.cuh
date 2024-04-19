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

#define CALC_POLY_FACE_DOT(result, vvLineV, vStartIndex, vEndIndex, polyOneV, polyN) \
    do { \
        result[0] = (vvLineV[vStartIndex*3 + AX] - polyOneV[AX]) * polyN[AX] + \
            (vvLineV[vStartIndex*3 + AY] - polyOneV[AY]) * polyN[AY] + \
            (vvLineV[vStartIndex*3 + AZ] - polyOneV[AZ]) * polyN[AZ]; \
        result[1] = (vvLineV[vEndIndex*3 + AX] - polyOneV[AX]) * polyN[AX] + \
            (vvLineV[vEndIndex*3 + AY] - polyOneV[AY]) * polyN[AY] + \
            (vvLineV[vEndIndex*3 + AZ] - polyOneV[AZ]) * polyN[AZ]; \
    } while(0);

#define CALC_VV_FACE_DOT(result, polyLineStartV, polyLineEndV, vvOneV, vvOneVIndex, vvN, vvNIndex) \
    do { \
        result[0] = \
        (polyLineStartV[AX] - vvOneV[vvOneVIndex + AX]) * vvN[vvNIndex + AX] + \
        (polyLineStartV[AY] - vvOneV[vvOneVIndex + AY]) * vvN[vvNIndex + AY] + \
        (polyLineStartV[AZ] - vvOneV[vvOneVIndex + AZ]) * vvN[vvNIndex + AZ]; \
        result[1] = \
        (polyLineEndV[AX] - vvOneV[vvOneVIndex + AX]) * vvN[vvNIndex + AX] + \
        (polyLineEndV[AY] - vvOneV[vvOneVIndex + AY]) * vvN[vvNIndex + AY] + \
        (polyLineEndV[AZ] - vvOneV[vvOneVIndex + AZ]) * vvN[vvNIndex + AZ]; \
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
    Render();

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
    std::vector<float> hMtCamTransRot;
    std::vector<float> hMtCamRot;

    std::vector<float> hCamViewAngleCos;

    int* hObjInJudgeAry;

};


#endif RENDER_CUH_