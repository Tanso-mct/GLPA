#include "render.cuh"

#define JUDGE_POLY_V_IN_VIEW_VOLUME(cnvt_poly_v, cam_far_z, cam_near_z, cam_view_angle, poly_v_in_if) \
{ \
    float cnvtXzPolyV[3] = {cnvt_poly_v[AX], 0, cnvt_poly_v[AZ]};\
    float cnvtYzPolyV[3] = {0, cnvt_poly_v[AY], cnvt_poly_v[AZ]};\
    \
    float zVec[3] = {0, 0, -1};\
    \
    float cnvtXzPolyVxZVecDotCos;\
    VEC_GET_VECS_COS(zVec, cnvtXzPolyV, cnvtXzPolyVxZVecDotCos);\
    \
    float cnvtYzPolyVxZVecDotCos;\
    VEC_GET_VECS_COS(zVec, cnvtYzPolyV, cnvtYzPolyVxZVecDotCos);\
    \
    int polyVZInIF = (cnvt_poly_v[AZ] >= -cam_far_z && cnvt_poly_v[AZ] <= -cam_near_z) ? TRUE : FALSE;\
    int polyXzVInIF = (cnvtXzPolyVxZVecDotCos >= cam_view_angle[AX]) ? TRUE : FALSE;\
    int polyYzVInIF = (cnvtYzPolyVxZVecDotCos >= cam_view_angle[AY]) ? TRUE : FALSE;\
    \
    poly_v_in_if = (polyVZInIF == TRUE && polyXzVInIF == TRUE && polyYzVInIF == TRUE) ? TRUE : FALSE;\
};

#define CALC_POLY_FACE_DOT(result, vv_line_v, v_start_index, v_end_index, poly_one_v, poly_n) \
{ \
    result[0] = (vv_line_v[v_start_index*3 + AX] - poly_one_v[AX]) * poly_n[AX] + \
        (vv_line_v[v_start_index*3 + AY] - poly_one_v[AY]) * poly_n[AY] + \
        (vv_line_v[v_start_index*3 + AZ] - poly_one_v[AZ]) * poly_n[AZ];\
    result[1] = (vv_line_v[v_end_index*3 + AX] - poly_one_v[AX]) * poly_n[AX] + \
        (vv_line_v[v_end_index*3 + AY] - poly_one_v[AY]) * poly_n[AY] + \
        (vv_line_v[v_end_index*3 + AZ] - poly_one_v[AZ]) * poly_n[AZ];\
};

#define CALC_VV_FACE_DOT(result, poly_line_start_v, poly_line_end_v, vv_one_v, vv_one_v_index, vv_n, vv_n_index) \
{ \
    result[0] = \
    (poly_line_start_v[AX] - vv_one_v[vv_one_v_index + AX]) * vv_n[vv_n_index + AX] + \
    (poly_line_start_v[AY] - vv_one_v[vv_one_v_index + AY]) * vv_n[vv_n_index + AY] + \
    (poly_line_start_v[AZ] - vv_one_v[vv_one_v_index + AZ]) * vv_n[vv_n_index + AZ];\
    result[1] = \
    (poly_line_end_v[AX] - vv_one_v[vv_one_v_index + AX]) * vv_n[vv_n_index + AX] + \
    (poly_line_end_v[AY] - vv_one_v[vv_one_v_index + AY]) * vv_n[vv_n_index + AY] + \
    (poly_line_end_v[AZ] - vv_one_v[vv_one_v_index + AZ]) * vv_n[vv_n_index + AZ];\
} 

#define CALC_VEC_COS(result, start_vec_1, end_vec_1, start_vec_2, end_vec_2) \
{ \
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
}

#define CALC_VEC_ARY_COS(result, start_vec_1, start_vec_1_index, end_vec_1, end_vec_1_index, start_vec_2, start_vec_2_index, end_vec_2, end_vec_2_index) \
{ \
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
}

#define VX_SCREEN_PIXEL_CONVERT(result, world_v, world_v_index, camera_near_z, near_screen_size, screen_pixel_size) \
{ \
    result = \
    std::round((((world_v[world_v_index + AX] * -camera_near_z / world_v[world_v_index + AZ]) + near_screen_size[AX] / 2) / \
    (near_screen_size[AX])) * screen_pixel_size[AX]);\
}

#define VY_SCREEN_PIXEL_CONVERT(result, world_v, world_v_index, camera_near_z, near_screen_size, screen_pixel_size) \
{ \
    result = \
    std::round(screen_pixel_size[AY] - (((world_v[world_v_index + AY] * -camera_near_z / world_v[world_v_index + AZ]) + near_screen_size[AY] / 2) / \
    (near_screen_size[AY])) * screen_pixel_size[AY]);\
}

#define JUDGE_V_ON_POLY_FACE(result, result_index, inxtn_amount, face_dot, line_i, view_volume_vs, vv_line_v_index, poly_vec_1, poly_vec_2, poly_vec_3, nearZ, near_sc_size, sc_pixel_size) \
{ \
    int vOnFaceIF = (face_dot == 0) ? TRUE : FALSE;\
    for (int conditionalBranch3; conditionalBranch3 < vOnFaceIF; conditionalBranch3++) \
    { \
        float inxtn[3] = { \
            view_volume_vs[vv_line_v_index*3 + AX], \
            view_volume_vs[vv_line_v_index*3 + AY], \
            view_volume_vs[vv_line_v_index*3 + AZ] \
        };\
        \
        float vecCos[6];\
        CALC_VEC_COS(vecCos[0], poly_vec_1, poly_vec_2, poly_vec_1, inxtn);\
        CALC_VEC_COS(vecCos[1], poly_vec_1, poly_vec_2, poly_vec_1, poly_vec_3);\
        \
        CALC_VEC_COS(vecCos[2], poly_vec_2, poly_vec_3, poly_vec_2, inxtn);\
        CALC_VEC_COS(vecCos[3], poly_vec_2, poly_vec_3, poly_vec_2, poly_vec_1);\
        \
        CALC_VEC_COS(vecCos[4], poly_vec_3, poly_vec_1, poly_vec_3, inxtn);\
        CALC_VEC_COS(vecCos[5], poly_vec_3, poly_vec_1, poly_vec_3, poly_vec_2);\
        \
        int inxtnInPolyFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch3 < inxtnInPolyFaceIF; conditionalBranch3++) \
        { \
            inxtn_amount += 1;\
            VX_SCREEN_PIXEL_CONVERT(result[result_index + AX], view_volume_vs, vv_line_v_index*3, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[result_index + AY], view_volume_vs, vv_line_v_index*3, nearZ, near_sc_size, sc_pixel_size);\
            result[result_index + AZ] = view_volume_vs[vv_line_v_index*3 + AZ];\
        } \
    } \
}

#define GET_POLY_ON_FACE_INXTN(result, line_index, inxtn_amount, face_dot, view_volume_vs, vv_line_index_1, vv_line_index_2, poly_vec_1, poly_vec_2, poly_vec_3, nearZ, near_sc_size, sc_pixel_size) \
{ \
    int calcInxtnIF = ((face_dot[0] > 0 && face_dot[1] < 0) || (face_dot[0] < 0 && face_dot[1] > 0)) ? TRUE : FALSE;\
    for(int conditionalBranch3 = 0; conditionalBranch3 < calcInxtnIF; conditionalBranch3++) \
    { \
        float inxtn[3];\
        for (int roopCoord = 0; roopCoord < 3; roopCoord++) \
        { \
            inxtn[roopCoord] = view_volume_vs[vv_line_index_1*3 + roopCoord] + \
                (view_volume_vs[vv_line_index_2*3 + roopCoord] - view_volume_vs[vv_line_index_1*3 + roopCoord]) * \
                (fabs(face_dot[0]) / (fabs(face_dot[0]) + fabs(face_dot[1])));\
        } \
        \
        float vecCos[6];\
        CALC_VEC_COS(vecCos[0], poly_vec_1, poly_vec_2, poly_vec_1, inxtn);\
        CALC_VEC_COS(vecCos[1], poly_vec_1, poly_vec_2, poly_vec_1, poly_vec_3);\
        CALC_VEC_COS(vecCos[2], poly_vec_2, poly_vec_3, poly_vec_2, inxtn);\
        CALC_VEC_COS(vecCos[3], poly_vec_2, poly_vec_3, poly_vec_2, poly_vec_1);\
        CALC_VEC_COS(vecCos[4], poly_vec_3, poly_vec_1, poly_vec_3, inxtn);\
        CALC_VEC_COS(vecCos[5], poly_vec_3, poly_vec_1, poly_vec_3, poly_vec_2);\
        \
        int inxtnInPolyFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch4 < inxtnInPolyFaceIF; conditionalBranch4++) \
        { \
            inxtn_amount += 1;\
            VX_SCREEN_PIXEL_CONVERT(result[line_index + AX], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[line_index + AY], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            result[line_index + AZ] = inxtn[AZ];\
        } \
    } \
}

#define JUDGE_V_ON_VV_FACE(result, result_index, inxtn_amount, face_dot, poly_v, face_i, view_volume_vs, vv_face_vs_index, nearZ, near_sc_size, sc_pixel_size) \
{ \
    int vOnFaceIF = (face_dot == 0) ? TRUE : FALSE;\
    for (int conditionalBranch3; conditionalBranch3 < vOnFaceIF; conditionalBranch3++) \
    { \
        float inxtn[3] = {poly_v[AX], poly_v[AY], poly_v[AZ]};\
        \
        float vecCos[8];\
        CALC_VEC_ARY_COS(vecCos[0], view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[1], view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[2], view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[3], view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[4], view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[5], view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V2]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[6], view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[7], view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_i*4 + V3]*3);\
        \
        int inxtnInVvFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5] && vecCos[6] >= vecCos[7]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch3 < inxtnInVvFaceIF; conditionalBranch3++) \
        { \
            inxtn_amount += 1;\
            VX_SCREEN_PIXEL_CONVERT(result[result_index + AX], poly_v, 0, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[result_index + AY], poly_v, 0, nearZ, near_sc_size, sc_pixel_size);\
            result[result_index + AZ] = poly_v[AZ];\
        } \
    } \
}

#define GET_POLY_ON_LINE_INXTN(result, result_index, inxtn_amount, poly_line_v1, poly_line_v2, face_dot, view_volume_vs, vv_face_vs_index, face_index, nearZ, near_sc_size, sc_pixel_size) \
{ \
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
        CALC_VEC_ARY_COS(vecCos[0], view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[1], view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[2], view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[3], view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[4], view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[5], view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V2]*3);\
        \
        CALC_VEC_ARY_COS(vecCos[6], view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, inxtn, 0);\
        CALC_VEC_ARY_COS(vecCos[7], view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V1]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V4]*3, view_volume_vs, vv_face_vs_index[face_index*4 + V3]*3);\
        \
        int inxtnInVvFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5] && vecCos[6] >= vecCos[7]) ? TRUE : FALSE;\
        \
        for (int conditionalBranch4 = 0; conditionalBranch3 < inxtnInVvFaceIF; conditionalBranch3++) \
        { \
            inxtn_amount += 1;\
            VX_SCREEN_PIXEL_CONVERT(result[result_index + AX], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            VY_SCREEN_PIXEL_CONVERT(result[result_index + AY], inxtn, 0, nearZ, near_sc_size, sc_pixel_size);\
            result[result_index + AZ] = inxtn[AZ];\
        } \
    } \
}

Render::Render()
{
    hMtCamTransRot = std::vector<float>(16);
    hMtCamRot = std::vector<float>(16);

    hCamViewAngleCos = std::vector<float>(2);

}

__global__ void glpaGpuPrepareObj(
    int objSize,
    float* objWVs,
    float* mtCamTransRot,
    float camNearZ,
    float camFarZ,
    float* camViewAngleCos,
    int* result
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < objSize)
    {
        int objRectStatus = 0;
        float objRectOrigin[3];
        float objRectOpposite[3];
        for (int objWvsI = 0; objWvsI < 8; objWvsI++)
        {
            float vec3d[3] = {objWVs[i*8*3 + objWvsI*3 + AX], objWVs[i*8*3 + objWvsI*3 + AY], objWVs[i*8*3 + objWvsI*3 + AZ]};

            float camObjVs[3] = {
                vec3d[AX] * mtCamTransRot[0] + vec3d[AY] * mtCamTransRot[1] + vec3d[AZ] * mtCamTransRot[2] + 1 * mtCamTransRot[3],
                vec3d[AX] * mtCamTransRot[4] + vec3d[AY] * mtCamTransRot[5] + vec3d[AZ] * mtCamTransRot[6] + 1 * mtCamTransRot[7],
                vec3d[AX] * mtCamTransRot[8] + vec3d[AY] * mtCamTransRot[9] + vec3d[AZ] * mtCamTransRot[10] + 1 * mtCamTransRot[11]
            };

            int objRectStatusIF = (objRectStatus > 0) ? TRUE : FALSE;

            objRectOrigin[AX] = (objRectStatusIF == FALSE) ? camObjVs[AX] : (camObjVs[AX] < objRectOrigin[AX]) ? camObjVs[AX] : objRectOrigin[AX];
            objRectOrigin[AY] = (objRectStatusIF == FALSE) ? camObjVs[AY] : (camObjVs[AY] < objRectOrigin[AY]) ? camObjVs[AY] : objRectOrigin[AY];
            objRectOrigin[AZ] = (objRectStatusIF == FALSE) ? camObjVs[AZ] : (camObjVs[AZ] > objRectOrigin[AZ]) ? camObjVs[AZ] : objRectOrigin[AZ];

            objRectOpposite[AX] = (objRectStatusIF == FALSE) ? camObjVs[AX] : (camObjVs[AX] > objRectOpposite[AX]) ? camObjVs[AX] : objRectOpposite[AX];
            objRectOpposite[AY] = (objRectStatusIF == FALSE) ? camObjVs[AY] : (camObjVs[AY] > objRectOpposite[AY]) ? camObjVs[AY] : objRectOpposite[AY];
            objRectOpposite[AZ] = (objRectStatusIF == FALSE) ? camObjVs[AZ] : (camObjVs[AZ] < objRectOpposite[AZ]) ? camObjVs[AZ] : objRectOpposite[AZ];

            objRectStatus += 1;

        }

        float objOppositeVs[12] = {
            objRectOrigin[AX], 0, objRectOpposite[AZ],
            objRectOpposite[AX], 0, objRectOpposite[AZ],
            0, objRectOrigin[AY], objRectOpposite[AZ],
            0, objRectOpposite[AY], objRectOpposite[AZ]
        };


        float zVec[3] = {0, 0, -1};
        float vecsCos[4];

        for (int aryI = 0; aryI < 4; aryI++)
        {
            float calcObjOppositeV[3] = {
                objOppositeVs[aryI*3 + AX],
                objOppositeVs[aryI*3 + AY],
                objOppositeVs[aryI*3 + AZ]
            };

            VEC_GET_VECS_COS(zVec, calcObjOppositeV, vecsCos[aryI]);
        }

        int objZInIF = (objRectOrigin[AZ] >= -camFarZ && objRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;
        int objXzInIF = (vecsCos[0] >= camViewAngleCos[AX] || vecsCos[1] >= camViewAngleCos[AX]) ? TRUE : FALSE;
        int objYzInIF = (vecsCos[2] >= camViewAngleCos[AY] || vecsCos[3] >= camViewAngleCos[AY]) ? TRUE : FALSE;

        int objInIF = (objZInIF == TRUE && objXzInIF == TRUE && objYzInIF == TRUE) ? i + 1 : 0;

        result[objInIF] = TRUE;
    }
}

void Render::prepareObjs(std::unordered_map<std::wstring, Object> sObj, Camera cam)
{
    std::vector<float> hObjWvs;

    for (auto obj : sObj)
    {
        for (int i = 0; i < 8; i++)
        {
            hObjWvs.push_back(obj.second.range.wVertex[i].x / CALC_SCALE);
            hObjWvs.push_back(obj.second.range.wVertex[i].y / CALC_SCALE);
            hObjWvs.push_back(obj.second.range.wVertex[i].z / CALC_SCALE);
        }
    }

    float* dObjWvs;
    cudaMalloc((void**)&dObjWvs, sizeof(float)*hObjWvs.size());
    cudaMemcpy(dObjWvs, hObjWvs.data(), sizeof(float)*hObjWvs.size(), cudaMemcpyHostToDevice);

    hMtCamTransRot[0] = cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamTransRot[1] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[2] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[3] = -cam.wPos.x / CALC_SCALE;
    hMtCamTransRot[4] = sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamTransRot[5] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[6] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[7] = -cam.wPos.y / CALC_SCALE;
    hMtCamTransRot[8] = -sin(RAD(-cam.rotAngle.y));
    hMtCamTransRot[9] = cos(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[10] = cos(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[11] = -cam.wPos.z / CALC_SCALE;
    hMtCamTransRot[12] = 0;
    hMtCamTransRot[13] = 0;
    hMtCamTransRot[14] = 0;
    hMtCamTransRot[15] = 1;

    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*hMtCamTransRot.size());
    cudaMemcpy(dMtCamTransRot, hMtCamTransRot.data(), sizeof(float)*hMtCamTransRot.size(), cudaMemcpyHostToDevice);

    hCamViewAngleCos[AX] = cam.viewAngleCos.x;
    hCamViewAngleCos[AY] = cam.viewAngleCos.y;

    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*hCamViewAngleCos.size());
    cudaMemcpy(dCamViewAngleCos, hCamViewAngleCos.data(), sizeof(float)*hCamViewAngleCos.size(), cudaMemcpyHostToDevice);


    hObjInJudgeAry = new int[sObj.size() + 1];
    std::fill(hObjInJudgeAry, hObjInJudgeAry + sObj.size() + 1, FALSE); 

    int* dObjInJudgeAry;
    cudaMalloc((void**)&dObjInJudgeAry, sizeof(int)*(sObj.size() + 1));


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = sObj.size();
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    glpaGpuPrepareObj<<<dimGrid, dimBlock>>>
    (
        sObj.size(),
        dObjWvs,
        dMtCamTransRot,
        static_cast<float>(cam.nearZ) / CALC_SCALE,
        static_cast<float>(cam.farZ) / CALC_SCALE,
        dCamViewAngleCos,
        dObjInJudgeAry
    );

    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }

    cudaMemcpy(hObjInJudgeAry, dObjInJudgeAry, sizeof(int)*(sObj.size() + 1), cudaMemcpyDeviceToHost);

    cudaFree(dObjWvs);
    cudaFree(dMtCamTransRot);
    cudaFree(dCamViewAngleCos);
    cudaFree(dObjInJudgeAry);


}

__global__ void glpaGpuRender(
    float* polyVs,
    float* polyNs,
    int polyAmount,
    float* mtCamTransRot,
    float* mtCamRot,
    float camFarZ,
    float camNearZ,
    float* camViewAngleCos,
    float* viewVolumeVs,
    float* viewVolumeNs,
    float* nearScSize,
    float* scPixelSize
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyAmount)
    {
        float vec3d[3];

        vec3d[AX] = polyVs[i*9 + AX];
        vec3d[AY] = polyVs[i*9 + AY];
        vec3d[AZ] = polyVs[i*9 + AZ];
        float cnvtPolyV1[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV1);

        vec3d[AX] = polyVs[i*9 + 3 + AX];
        vec3d[AY] = polyVs[i*9 + 3 + AY];
        vec3d[AZ] = polyVs[i*9 + 3 + AZ];
        float cnvtPolyV2[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV2);

        vec3d[AX] = polyVs[i*9 + 6 + AX];
        vec3d[AY] = polyVs[i*9 + 6 + AY];
        vec3d[AZ] = polyVs[i*9 + 6 + AZ];
        float cnvtPolyV3[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV3);

        vec3d[AX] = polyNs[i*3 + AX];
        vec3d[AY] = polyNs[i*3 + AY];
        vec3d[AZ] = polyNs[i*3 + AZ];
        float cnvtPolyN[3];
        MT_PRODUCT_4X4_VEC3D(mtCamRot, vec3d, cnvtPolyN);

        float polyVxPolyNDotCos;
        VEC_GET_VECS_COS(cnvtPolyN, cnvtPolyV1, polyVxPolyNDotCos);
        
        int polyBilateralIF = (polyVxPolyNDotCos <= 0) ? TRUE : FALSE;

        for (int conditionalBranch = 0; conditionalBranch < polyBilateralIF; conditionalBranch++)
        {
            int polyV1InIF;
            int polyV2InIF;
            int polyV3InIF;
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV1, camFarZ, camNearZ, camViewAngleCos, polyV1InIF);
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV2, camFarZ, camNearZ, camViewAngleCos, polyV2InIF);
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV3, camFarZ, camNearZ, camViewAngleCos, polyV3InIF);

            int noVsInIF = (polyV1InIF == FALSE && polyV2InIF == FALSE && polyV3InIF == FALSE) ? TRUE : FALSE;

            int shapeCnvtIF = (polyV1InIF + polyV2InIF + polyV3InIF != 3) ? TRUE : FALSE;

            int polyInIF = (polyV1InIF == TRUE || polyV2InIF == TRUE || polyV3InIF == TRUE) ? TRUE : FALSE;
            for (int conditionalBranch2 = 0; conditionalBranch2 < noVsInIF; conditionalBranch2++)
            {
                float polyRectOrigin[3] = {cnvtPolyV1[AX], cnvtPolyV1[AY], cnvtPolyV1[AZ]};
                float polyRectOpposite[3] = {cnvtPolyV1[AX], cnvtPolyV1[AY], cnvtPolyV1[AZ]};

                polyRectOrigin[AX] = (cnvtPolyV2[AX] < polyRectOrigin[AX]) ? cnvtPolyV2[AX] : polyRectOrigin[AX];
                polyRectOrigin[AY] = (cnvtPolyV2[AY] < polyRectOrigin[AY]) ? cnvtPolyV2[AY] : polyRectOrigin[AY];
                polyRectOrigin[AZ] = (cnvtPolyV2[AZ] > polyRectOrigin[AZ]) ? cnvtPolyV2[AZ] : polyRectOrigin[AZ];

                polyRectOpposite[AX] = (cnvtPolyV2[AX] > polyRectOpposite[AX]) ? cnvtPolyV2[AX] : polyRectOpposite[AX];
                polyRectOpposite[AY] = (cnvtPolyV2[AY] > polyRectOpposite[AY]) ? cnvtPolyV2[AY] : polyRectOpposite[AY];
                polyRectOpposite[AZ] = (cnvtPolyV2[AZ] < polyRectOpposite[AZ]) ? cnvtPolyV2[AZ] : polyRectOpposite[AZ];

                polyRectOrigin[AX] = (cnvtPolyV3[AX] < polyRectOrigin[AX]) ? cnvtPolyV3[AX] : polyRectOrigin[AX];
                polyRectOrigin[AY] = (cnvtPolyV3[AY] < polyRectOrigin[AY]) ? cnvtPolyV3[AY] : polyRectOrigin[AY];
                polyRectOrigin[AZ] = (cnvtPolyV3[AZ] > polyRectOrigin[AZ]) ? cnvtPolyV3[AZ] : polyRectOrigin[AZ];

                polyRectOpposite[AX] = (cnvtPolyV3[AX] > polyRectOpposite[AX]) ? cnvtPolyV3[AX] : polyRectOpposite[AX];
                polyRectOpposite[AY] = (cnvtPolyV3[AY] > polyRectOpposite[AY]) ? cnvtPolyV3[AY] : polyRectOpposite[AY];
                polyRectOpposite[AZ] = (cnvtPolyV3[AZ] < polyRectOpposite[AZ]) ? cnvtPolyV3[AZ] : polyRectOpposite[AZ];

                // TODO: 3 and 4 are different from the source. This may be the cause of the bug, so please check.
                float polyRectOppositeSideVs[12] = {
                    polyRectOrigin[AX], 0,  polyRectOpposite[AZ],
                    polyRectOpposite[AX], 0, polyRectOpposite[AZ],
                    0, polyRectOrigin[AY], polyRectOpposite[AZ],
                    0, polyRectOpposite[AY], polyRectOpposite[AZ]
                };

                float zVec[3] = {0, 0, -1};
                float vecsCos[4];

                for (int aryI = 0; aryI < 4; aryI++)
                {
                    float calcObjOppositeV[3] = {
                        polyRectOppositeSideVs[aryI*3 + AX],
                        polyRectOppositeSideVs[aryI*3 + AY],
                        polyRectOppositeSideVs[aryI*3 + AZ]
                    };

                    VEC_GET_VECS_COS(zVec, calcObjOppositeV, vecsCos[aryI]);
                }

                int polyZInIF = (polyRectOrigin[AZ] >= -camFarZ && polyRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;
                int polyXzInIF = (vecsCos[0] >= camViewAngleCos[AX] || vecsCos[1] >= camViewAngleCos[AX]) ? TRUE : FALSE;
                int polyYzInIF = (vecsCos[2] >= camViewAngleCos[AY] || vecsCos[3] >= camViewAngleCos[AY]) ? TRUE : FALSE;

                polyInIF = (polyZInIF == TRUE && polyXzInIF == TRUE && polyYzInIF == TRUE) ? TRUE : FALSE;
            }

            for(int conditionalBranch2 = 0; conditionalBranch2 < polyInIF; conditionalBranch2++)
            {
                int vvFaceI[6] = {
                    RECT_FRONT_TOP_LEFT,
                    RECT_FRONT_TOP_LEFT,
                    RECT_BACK_BOTTOM_RIGHT,
                    RECT_FRONT_TOP_LEFT,
                    RECT_BACK_BOTTOM_RIGHT,
                    RECT_BACK_BOTTOM_RIGHT
                };

                int vvFaceVsI[24] = {
                    VIEWVOLUME_TOP_V1, VIEWVOLUME_TOP_V2, VIEWVOLUME_TOP_V3, VIEWVOLUME_TOP_V4,
                    VIEWVOLUME_FRONT_V1, VIEWVOLUME_FRONT_V2, VIEWVOLUME_FRONT_V3, VIEWVOLUME_FRONT_V4,
                    VIEWVOLUME_RIGHT_V1, VIEWVOLUME_RIGHT_V2, VIEWVOLUME_RIGHT_V3, VIEWVOLUME_RIGHT_V4,
                    VIEWVOLUME_LEFT_V1, VIEWVOLUME_LEFT_V2, VIEWVOLUME_LEFT_V3, VIEWVOLUME_LEFT_V4,
                    VIEWVOLUME_BACK_V1, VIEWVOLUME_BACK_V2, VIEWVOLUME_BACK_V3, VIEWVOLUME_BACK_V4,
                    VIEWVOLUME_BOTTOM_V1, VIEWVOLUME_BOTTOM_V2, VIEWVOLUME_BOTTOM_V3, VIEWVOLUME_BOTTOM_V4
                };

                int vvLineVI[24] = {
                    RECT_L1_STARTV, RECT_L1_ENDV,
                    RECT_L2_STARTV, RECT_L2_ENDV,
                    RECT_L3_STARTV, RECT_L3_ENDV,
                    RECT_L4_STARTV, RECT_L4_ENDV,
                    RECT_L5_STARTV, RECT_L5_ENDV,
                    RECT_L6_STARTV, RECT_L6_ENDV,
                    RECT_L7_STARTV, RECT_L7_ENDV,
                    RECT_L8_STARTV, RECT_L8_ENDV,
                    RECT_L9_STARTV, RECT_L9_ENDV,
                    RECT_L10_STARTV, RECT_L10_ENDV,
                    RECT_L11_STARTV, RECT_L11_ENDV,
                    RECT_L12_STARTV, RECT_L12_ENDV
                };

                int inxtnAmount = 0;

                float pixelFaceInxtn[12 * 3 * 3] = {FALSE};
                for (int roopLineI = 0; roopLineI < 12; roopLineI++)
                {
                    float polyFaceDot[2];
                    CALC_POLY_FACE_DOT(polyFaceDot, viewVolumeVs, vvLineVI[roopLineI*2], vvLineVI[roopLineI*2 + 1], cnvtPolyV1, cnvtPolyN);

                    JUDGE_V_ON_POLY_FACE(
                        pixelFaceInxtn, roopLineI*3*3, inxtnAmount, polyFaceDot[0], roopLineI, viewVolumeVs, vvLineVI[roopLineI*2], 
                        cnvtPolyV1, cnvtPolyV2, cnvtPolyV3, camNearZ, nearScSize, scPixelSize
                    );
                    JUDGE_V_ON_POLY_FACE(
                        pixelFaceInxtn, roopLineI*3*3 + 6, inxtnAmount, polyFaceDot[1], roopLineI, viewVolumeVs, vvLineVI[roopLineI*2 + 1], 
                        cnvtPolyV1, cnvtPolyV2, cnvtPolyV3, camNearZ, nearScSize, scPixelSize
                    );

                    GET_POLY_ON_FACE_INXTN(
                        pixelFaceInxtn, roopLineI*3*3 + 9, inxtnAmount, polyFaceDot, viewVolumeNs, vvLineVI[roopLineI*2], vvLineVI[roopLineI*2 + 1], 
                        cnvtPolyV1, cnvtPolyV2, cnvtPolyV3, camNearZ, nearScSize, scPixelSize
                    );
                }

                float pixelLineInxtn[3*3 + 3*3] = {FALSE};
                for (int roopFaceI = 0; roopFaceI < 6; roopFaceI++)
                {
                    float vvFaceDot[2];
                    CALC_VV_FACE_DOT(vvFaceDot, cnvtPolyV1, cnvtPolyV2, viewVolumeVs, vvFaceI[roopFaceI], viewVolumeNs, roopFaceI);
                    JUDGE_V_ON_VV_FACE(
                        pixelLineInxtn, roopFaceI*3*9 + 0, inxtnAmount, vvFaceDot[0], cnvtPolyV1, roopFaceI, 
                        viewVolumeVs, vvFaceVsI, camNearZ, nearScSize, scPixelSize
                    );
                    JUDGE_V_ON_VV_FACE(
                        pixelLineInxtn, roopFaceI*3*9 + 3, inxtnAmount, vvFaceDot[1], cnvtPolyV2, roopFaceI, 
                        viewVolumeVs, vvFaceVsI, camNearZ, nearScSize, scPixelSize
                    );
                    GET_POLY_ON_LINE_INXTN(
                        pixelLineInxtn, roopFaceI*3*9 + 9, inxtnAmount, cnvtPolyV1, cnvtPolyV2, vvFaceDot, 
                        viewVolumeVs, vvFaceVsI, roopFaceI, camNearZ, nearScSize, scPixelSize
                    );

                    CALC_VV_FACE_DOT(vvFaceDot, cnvtPolyV2, cnvtPolyV3, viewVolumeVs, vvFaceI[roopFaceI], viewVolumeNs, roopFaceI);
                    JUDGE_V_ON_VV_FACE(
                        pixelLineInxtn, roopFaceI*3*9 + 6, inxtnAmount, vvFaceDot[1], cnvtPolyV3, roopFaceI, 
                        viewVolumeVs, vvFaceVsI, camNearZ, nearScSize, scPixelSize
                    );
                    GET_POLY_ON_LINE_INXTN(
                        pixelLineInxtn, roopFaceI*3*9 + 12, inxtnAmount, cnvtPolyV2, cnvtPolyV3, vvFaceDot, 
                        viewVolumeVs, vvFaceVsI, roopFaceI, camNearZ, nearScSize, scPixelSize
                    );

                    CALC_VV_FACE_DOT(vvFaceDot, cnvtPolyV3, cnvtPolyV1, viewVolumeVs, vvFaceI[roopFaceI], viewVolumeNs, roopFaceI);
                    GET_POLY_ON_LINE_INXTN(
                        pixelLineInxtn, roopFaceI*3*9 + 15, inxtnAmount, cnvtPolyV3, cnvtPolyV1, vvFaceDot, 
                        viewVolumeVs, vvFaceVsI, roopFaceI, camNearZ, nearScSize, scPixelSize
                    );
                }

                float pixelVs[(polyV1InIF + polyV2InIF + polyV3InIF + inxtnAmount)*3];

            }

        }

        


    }

}

void Render::render(std::unordered_map<std::wstring, Object> sObj, Camera cam, LPDWORD buffer)
{
    std::vector<float> polyVs;
    std::vector<float> polyNs;
    int loopObjI = 1;
    for (auto obj : sObj)
    {
        if (hObjInJudgeAry[loopObjI] == FALSE)
        {
            continue;
        }

        for (int i = 0; i < obj.second.poly.vId.size(); i++)
        {
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].z / CALC_SCALE);

            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].z / CALC_SCALE);

            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].z / CALC_SCALE);

            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].x);
            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].y);
            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].z);
        }
    }

    float* dPolyVs;
    float* dPolyNs;
    cudaMalloc((void**)&dPolyVs, sizeof(float)*polyVs.size());
    cudaMalloc((void**)&dPolyNs, sizeof(float)*polyNs.size());
    cudaMemcpy(dPolyVs, polyVs.data(), sizeof(float)*polyVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyVs, polyNs.data(), sizeof(float)*polyNs.size(), cudaMemcpyHostToDevice);


    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*hMtCamTransRot.size());
    cudaMemcpy(dMtCamTransRot, hMtCamTransRot.data(), sizeof(float)*hMtCamTransRot.size(), cudaMemcpyHostToDevice);


    hMtCamRot[0] = cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamRot[1] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[2] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamRot[3] = 0;
    hMtCamRot[4] = sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamRot[5] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[6] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamRot[7] = 0;
    hMtCamRot[8] = -sin(RAD(-cam.rotAngle.y));
    hMtCamRot[9] = cos(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x));
    hMtCamRot[10] = cos(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[11] = 0;
    hMtCamRot[12] = 0;
    hMtCamRot[13] = 0;
    hMtCamRot[14] = 0;
    hMtCamRot[15] = 1;

    float* dMtCamRot;
    cudaMalloc((void**)&dMtCamRot, sizeof(float)*hMtCamRot.size());
    cudaMemcpy(dMtCamRot, hMtCamRot.data(), sizeof(float)*hMtCamRot.size(), cudaMemcpyHostToDevice);


    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*hCamViewAngleCos.size());
    cudaMemcpy(dCamViewAngleCos, hCamViewAngleCos.data(), sizeof(float)*hCamViewAngleCos.size(), cudaMemcpyHostToDevice);


    std::vector<float> hViewVolumeVs;
    for (int i = 0; i < 8; i++){
        hViewVolumeVs.push_back(cam.viewVolume.v[i].x / CALC_SCALE);
        hViewVolumeVs.push_back(cam.viewVolume.v[i].y / CALC_SCALE);
        hViewVolumeVs.push_back(cam.viewVolume.v[i].z / CALC_SCALE);
    }

    float* dViewVolumeVs;
    cudaMalloc((void**)&dViewVolumeVs, sizeof(float)*hViewVolumeVs.size());
    cudaMemcpy(dViewVolumeVs, hViewVolumeVs.data(), sizeof(float)*hViewVolumeVs.size(), cudaMemcpyHostToDevice);


    std::vector<float> hViewVolumeNs;
    for (int i = 0; i < 6; i++){
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].x / CALC_SCALE);
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].y / CALC_SCALE);
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].z / CALC_SCALE);
    }

    float* dViewVolumeNs;
    cudaMalloc((void**)&dViewVolumeNs, sizeof(float)*hViewVolumeNs.size());
    cudaMemcpy(dViewVolumeNs, hViewVolumeNs.data(), sizeof(float)*hViewVolumeNs.size(), cudaMemcpyHostToDevice);


    std::vector<float> hNearScSize;
    hNearScSize.push_back(cam.nearScrSize.x / CALC_SCALE);
    hNearScSize.push_back(cam.nearScrSize.y / CALC_SCALE);

    float* dNearScSize;
    cudaMalloc((void**)&dNearScSize, sizeof(float)*hNearScSize.size());
    cudaMemcpy(dNearScSize, hNearScSize.data(), sizeof(float)*hNearScSize.size(), cudaMemcpyHostToDevice);

    
    std::vector<float> hScPixelSize;
    hScPixelSize.push_back(cam.scPixelSize.x / CALC_SCALE);
    hScPixelSize.push_back(cam.scPixelSize.y / CALC_SCALE);

    float* dScPixelSize;
    cudaMalloc((void**)&dScPixelSize, sizeof(float)*hScPixelSize.size());
    cudaMemcpy(dScPixelSize, hScPixelSize.data(), sizeof(float)*hScPixelSize.size(), cudaMemcpyHostToDevice);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = polyVs.size() / 9;
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    // glpaGpuRender<<<dimGrid, dimBlock>>>
    // (
        
    // );

    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }

    delete[] hObjInJudgeAry;






}
