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

#include "vector.cuh"
#include "matrix.cuh"

#define JUDGE_POLY_V_IN_VIEW_VOLUME(cnvt_poly_v, cam_far_z, cam_near_z, cam_view_angle, poly_v_in_if) \
do { \
    float cnvtXzPolyV[3] = {cnvt_poly_v[AX], 0, cnvt_poly_v[AZ]};\
    float cnvtYzPolyV[3] = {0, cnvt_poly_v[AY], cnvt_poly_v[AZ]};\
    \
    float zVec[3] = {0, 0, -1};\
    \
    float cnvtXzPolyVxZVecDotCos = 0;\
    VEC_GET_VECS_COS(zVec, cnvtXzPolyV, cnvtXzPolyVxZVecDotCos);\
    \
    float cnvtYzPolyVxZVecDotCos = 0;\
    VEC_GET_VECS_COS(zVec, cnvtYzPolyV, cnvtYzPolyVxZVecDotCos);\
    \
    int polyVZInIF = (cnvt_poly_v[AZ] >= -cam_far_z && cnvt_poly_v[AZ] <= -cam_near_z) ? TRUE : FALSE;\
    int polyXzVInIF = (cnvtXzPolyVxZVecDotCos >= cam_view_angle[AX]) ? TRUE : FALSE;\
    int polyYzVInIF = (cnvtYzPolyVxZVecDotCos >= cam_view_angle[AY]) ? TRUE : FALSE;\
    \
    poly_v_in_if = (polyVZInIF == TRUE && polyXzVInIF == TRUE && polyYzVInIF == TRUE) ? TRUE : FALSE;\
} while(0); \

#define CALC_VEC_COS(result, start_vec_1, end_vec_1, start_vec_2, end_vec_2) \
do { \
    result = \
    ((end_vec_1[AX] - start_vec_1[AX]) * (end_vec_2[AX] - start_vec_2[AX]) + \
    (end_vec_1[AY] - start_vec_1[AY]) * (end_vec_2[AY] - start_vec_2[AY]) + \
    (end_vec_1[AZ] - start_vec_1[AZ]) * (end_vec_2[AZ] - start_vec_2[AZ])) / \
    (sqrt((end_vec_1[AX] - start_vec_1[AX]) * (end_vec_1[AX] - start_vec_1[AX]) + \
    (end_vec_1[AY] - start_vec_1[AY]) * (end_vec_1[AY] - start_vec_1[AY]) + \
    (end_vec_1[AZ] - start_vec_1[AZ]) * (end_vec_1[AZ] - start_vec_1[AZ])) * \
    sqrt((end_vec_2[AX] - start_vec_2[AX]) * (end_vec_2[AX] - start_vec_2[AX]) + \
    (end_vec_2[AY] - start_vec_2[AY]) * (end_vec_2[AY] - start_vec_2[AY]) + \
    (end_vec_2[AZ] - start_vec_2[AZ]) * (end_vec_2[AZ] - start_vec_2[AZ])));\
} while(0); \

#define CALC_VEC_ARY_COS(result, start_vec_1, start_vec_1_index, end_vec_1, end_vec_1_index, start_vec_2, start_vec_2_index, end_vec_2, end_vec_2_index) \
do { \
    result = \
    ((end_vec_1[end_vec_1_index + AX] - start_vec_1[start_vec_1_index + AX]) * (end_vec_2[end_vec_2_index + AX] - start_vec_2[start_vec_2_index + AX]) + \
    (end_vec_1[end_vec_1_index + AY] - start_vec_1[start_vec_1_index + AY]) * (end_vec_2[end_vec_2_index + AY] - start_vec_2[start_vec_2_index + AY]) + \
    (end_vec_1[end_vec_1_index + AZ] - start_vec_1[start_vec_1_index + AZ]) * (end_vec_2[end_vec_2_index + AZ] - start_vec_2[start_vec_2_index + AZ])) / \
    (sqrt((end_vec_1[end_vec_1_index + AX] - start_vec_1[start_vec_1_index + AX]) * (end_vec_1[end_vec_1_index + AX] - start_vec_1[start_vec_1_index + AX]) + \
    (end_vec_1[end_vec_1_index + AY] - start_vec_1[start_vec_1_index + AY]) * (end_vec_1[end_vec_1_index + AY] - start_vec_1[start_vec_1_index + AY]) + \
    (end_vec_1[end_vec_1_index + AZ] - start_vec_1[start_vec_1_index + AZ]) * (end_vec_1[end_vec_1_index + AZ] - start_vec_1[start_vec_1_index + AZ])) * \
    sqrt((end_vec_2[end_vec_2_index + AX] - start_vec_2[start_vec_2_index + AX]) * (end_vec_2[end_vec_2_index + AX] - start_vec_2[start_vec_2_index + AX]) + \
    (end_vec_2[end_vec_2_index + AY] - start_vec_2[start_vec_2_index + AY]) * (end_vec_2[end_vec_2_index + AY] - start_vec_2[start_vec_2_index + AY]) + \
    (end_vec_2[end_vec_2_index + AZ] - start_vec_2[start_vec_2_index + AZ]) * (end_vec_2[end_vec_2_index + AZ] - start_vec_2[start_vec_2_index + AZ])));\
} while(0); \

#define VX_SCREEN_PIXEL_CONVERT(result, world_v, world_v_index, camera_near_z, near_screen_size, screen_pixel_size) \
do { \
    result = \
    std::round((((world_v[world_v_index + AX] * -camera_near_z / world_v[world_v_index + AZ]) + near_screen_size[AX] / 2) / \
    (near_screen_size[AX])) * screen_pixel_size[AX]);\
} while(0); \

#define VY_SCREEN_PIXEL_CONVERT(result, world_v, world_v_index, camera_near_z, near_screen_size, screen_pixel_size) \
do { \
    result = \
    std::round(screen_pixel_size[AY] - (((world_v[world_v_index + AY] * -camera_near_z / world_v[world_v_index + AZ]) + near_screen_size[AY] / 2) / \
    (near_screen_size[AY])) * screen_pixel_size[AY]);\
} while(0); \

#define JUDGE_V_ON_VV_FACE(result, result_index, currentIndex, face_dot, poly_v, face_i, view_volume_vs, vv_face_vs_index, nearZ, near_sc_size, sc_pixel_size) \
do { \
    int vOnFaceIF = (face_dot == 0) ? TRUE : FALSE;\
    for (int conditionalBranch3 = 0; conditionalBranch3 < vOnFaceIF; conditionalBranch3++) \
    { \
        float inxtn[3] = {poly_v[AX], poly_v[AY], poly_v[AZ]};\
        \
        float vecCos[8];\
        CALC_VEC_ARY_COS(vecCos[0], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[1], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[2], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[3], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[4], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[5], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V2]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[6], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[7], view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + FACE_V3]*3);\
        \
        int inxtnInVvFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5] && vecCos[6] >= vecCos[7]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch4 < inxtnInVvFaceIF; conditionalBranch4++) \
        { \
            VX_SCREEN_PIXEL_CONVERT(result[(result_index)*3 + AX], poly_v, 0, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[(result_index)*3 + AY], poly_v, 0, nearZ, near_sc_size, sc_pixel_size);\
            result[(result_index)*3 + AZ] = poly_v[AZ];\
            currentIndex++;\
        } \
    } \
} while(0); \

#define GET_POLY_ON_LINE_INXTN(result, result_index, currentIndex, poly_line_v1, poly_line_v2, face_dot, view_volume_vs, vv_face_vs_index, face_index, nearZ, near_sc_size, sc_pixel_size) \
do { \
    int calcInxtnIF  = ((face_dot[0] > 0 && face_dot[1] < 0) || (face_dot[0] < 0 && face_dot[1] > 0)) ? TRUE : FALSE;\
    for(int conditionalBranch3 = 0; conditionalBranch3 < calcInxtnIF; conditionalBranch3++) \
    { \
        float inxtn[3];\
        for (int roopCoord = 0; roopCoord < 3; roopCoord++) \
        { \
            inxtn[roopCoord] = poly_line_v1[roopCoord] + \
                (poly_line_v2[roopCoord] - poly_line_v1[roopCoord]) * \
                (fabs(face_dot[0]) / (fabs(face_dot[0]) + fabs(face_dot[1])));\
        } \
        \
        float vecCos[8];\
        CALC_VEC_ARY_COS(vecCos[0], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[1], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[2], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[3], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[4], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[5], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V2]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[6], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[7], view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + FACE_V3]*3);\
        \
        int inxtnInVvFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5] && vecCos[6] >= vecCos[7]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch4 < inxtnInVvFaceIF; conditionalBranch4++) \
        { \
            VX_SCREEN_PIXEL_CONVERT(result[(result_index)*3 + AX], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[(result_index)*3 + AY], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            result[(result_index)*3 + AZ] = inxtn[AZ];\
            currentIndex++;\
        } \
    } \
} while(0); \


__global__ void glpaGpuPrepareObj(
    int object_size,
    float* object_world_vs,
    float* matrix_camera_transformation_and_rotation,
    float camera_near_z,
    float camera_far_z,
    float* camera_view_angle_cos,
    int* result_object_in_view_volume_judge_ary
);


__global__ void glpaGpuSetVs(
    float* poly_vertices,
    float* poly_normals,
    int poly_amount,
    float* matrix_camera_transformation_and_rotation,
    float* matrix_camera_rotation,
    float camera_far_z,
    float camera_near_z,
    float* camera_view_angle,
    float* view_volume_vertices,
    float* view_volume_normals,
    float* near_screen_size,
    float* screen_pixel_size,
    int* inxtn_amout,
    float* sort_inxtn,
    float* sort_pixel_inxtn,
    int side_vertices_size,
    float* debugAry
);


class Render{
public :
    Render();

    void gpuRender(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam,
        LPDWORD buffer
    );

    std::vector<float> hMtCamTransRot;
    std::vector<float> hMtCamRot;

    std::vector<float> hCamViewAngleCos;

    int sideVsSize = 0;

private :
    void prepareObjs(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam
    );

    void setVs(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam
    );

    void rasterize(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam,
        LPDWORD buffer
    );

    int* hObjInJudgeAry;

};


#endif RENDER_CUH_