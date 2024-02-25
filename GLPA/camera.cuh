/**
 * @file Camera.h
 * @brief Describes a process related to a camera that exists in 3D
 * @author Tanso
 * @date 2023-10
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <math.h>
#include <string>
#include <unordered_map>

#include "cg.h"
#include "view_volume.cuh"

#include "object.h"

#include "matrix.cuh"
#include "vector.cuh"


#define GLPA_CAMERA_OBJ_WV_1(numCombN) \
    objects[needRangeVs[i].objName].v.world[ \
        objects[needRangeVs[i].objName].poly.vId[needRangeVs[i].polyId].numCombN] \

#define GLPA_CAMERA_POLY_NEED_RANGE_IFS(numCombN) \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).x < polyRange.origin.x){ \
        polyRange.origin.x = GLPA_CAMERA_OBJ_WV_1(numCombN).x; \
    } \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).y < polyRange.origin.y){ \
        polyRange.origin.y = GLPA_CAMERA_OBJ_WV_1(numCombN).y; \
    } \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).z > polyRange.origin.z){ \
        polyRange.origin.z = GLPA_CAMERA_OBJ_WV_1(numCombN).z; \
    } \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).x > polyRange.opposite.x){ \
        polyRange.opposite.x = GLPA_CAMERA_OBJ_WV_1(numCombN).x; \
    } \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).y > polyRange.opposite.y){ \
        polyRange.opposite.y = GLPA_CAMERA_OBJ_WV_1(numCombN).y; \
    } \
    if (GLPA_CAMERA_OBJ_WV_1(numCombN).z < polyRange.opposite.z){ \
        polyRange.opposite.z = GLPA_CAMERA_OBJ_WV_1(numCombN).z; \
    } 


__global__ void glpaGpuGetPolyVvDot(
    double* poly_face_dot,
    double* vv_face_dot,
    double* poly_one_vs,
    double* poly_ns,
    double* vv_line_start_vs,
    double* vv_line_end_vs,
    double* vv_one_vs,
    double* vv_ns,
    double* poly_line_start_vs,
    double* poly_line_end_vs,
    int poly_face_amout,
    int vv_line_amout,
    int vv_face_amout,
    int poly_line_amout
);


__global__ void glpaGpuCalcIntxn(
    double* poly_face_dot,
    double* vv_face_dot,
    double* vv_line_start_vs,
    double* vv_line_end_vs,
    double* poly_line_start_vs,
    double* poly_line_end_vs,
    int* polyFaceVvLineI,
    int* polyFaceI,
    int* polyLineVvFaceI,
    int* vvLineI,
    int intxnAmount,
    double* face_inxtn,
    double* line_inxtn,
    double* poly_dot,
    double* face_dot
);

/// @brief Has data related to the 3DCG camera.
class Camera{
public :
    void load(
        std::wstring arg_name,
        Vec3d arg_w_pos,
        Vec3d arg_rot_angle,
        double arg_near_z,
        double arg_far_z,
        double arg_view_angle,
        Vec2d arg_aspect_ratio
    );

    void defineViewVolume();

    // void updateObjRectRange();

    void objCulling(std::unordered_map<std::wstring, Object> objects);

    void polyBilateralJudge(std::unordered_map<std::wstring, Object> objects);

    void polyCulling(
        std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* pt_rasterize_source
    );

    void polyShapeConvert(
        std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* pt_rasterize_source
    );

    Matrix mt;
    Vector vec;

private : 
    bool reload = false;

    std::wstring name = GLPA_WSTRING_DEF;
    Vec3d wPos = {0, 0, 0};
    Vec3d rotAngle = {0, 0, 0};

    double nearZ = -1;
    double farZ = -10000;
    double viewAngle = 80;
    Vec2d viewAngleCos;
    Vec2d aspectRatio = {16, 9};

    Vec2d nearScrSize;
    Vec2d farScrSize;

    ViewVolume viewVolume;
    
    std::vector<std::wstring> renderTargetObj;
    std::vector<PolyNameInfo> renderTargetPoly;

    std::vector<int> shapeCnvtTargetI;
    std::vector<int> calcIntxnTargetI;

    
};

#endif CAMERA_H_
