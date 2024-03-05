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
#include <cmath>
#include <string>
#include <unordered_map>
#include <Windows.h>

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


__global__ void glpaGpuGetIntxn(
    double* poly_faceLine_vs,
    double* poly_face_dot,
    double* poly_face_inxtn,
    double* vv_faceLine_vs,
    double* vv_face_dot,
    double* vv_face_inxtn,
    int poly_face_size,
    int vv_face_size
);


__global__ void glpaGpuGetIACos(
    double* poly_face_vs,
    double* poly_face_inxtn,
    double* poly_face_ia_cos,
    double* vv_face_vs,
    double* vv_face_inxtn,
    double* vv_face_ia_cos,
    int poly_face_size,
    int vv_face_size
);


__global__ void glpaGpuScPixelConvert(
    double* world_vs,
    double* near_z,
    double* far_z,
    double* near_screen_size,
    double* screen_pixel_size,
    double* result_vs,
    int world_vs_amount
);


__global__ void glpaGpuSortVsDotCross(
    int* sort_vs_sizes,
    double* sort_vs,
    double* dot_cos,
    double* cross,
    int target_size
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
        Vec2d arg_aspect_ratio,
        Vec2d arg_screen_pixel_size
    );

    void defineViewVolume();

    // void updateObjRectRange();

    void objCulling(std::unordered_map<std::wstring, Object> objects);

    void polyBilateralJudge(std::unordered_map<std::wstring, Object> objects);

    void polyCulling(
        std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* pt_rasterize_source
    );

    void polyVvLineDot(
        std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* pt_rasterize_source
    );

    void pushLineToV(
        std::vector<Vec3d> source_vs,
        std::vector<double>* target_vs,
        int start_i,
        int end_i
    );

    void inxtnInteriorAngle(std::vector<RasterizeSource>* pt_rasterize_source);

    void setPolyInxtn(
        std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* pt_rasterize_source
    );

    void scPixelConvert(std::vector<RasterizeSource>* pt_rasterize_source);

    void sortScPixelVs(std::vector<RasterizeSource>* pt_rasterize_source);

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
    Vec2d viewAngleTan;
    Vec2d aspectRatio = {16, 9};

    Vec2d nearScrSize;
    Vec2d farScrSize;
    Vec2d scPixelSize;

    ViewVolume viewVolume;
    
    std::vector<std::wstring> renderTargetObj;
    std::vector<PolyNameInfo> renderTargetPoly;

    std::vector<int> shapeCnvtTargetI;

    std::vector<int> sort4TargetI;
    std::vector<int> sort5TargetI;
    std::vector<int> sort6TargetI;
    std::vector<int> sort7TargetI;

    std::vector<double> sort4Vs;
    std::vector<double> sort5Vs;
    std::vector<double> sort6Vs;
    std::vector<double> sort7Vs;

    double* hPolyFaceDot;
    double* hVvFaceDot;

    std::vector<int> polyRsI;
    std::vector<int> vvRsI;

    double* hPolyFaceInxtn;
    double* hVvFaceInxtn;

    double* hPolyFaceIACos;
    double* hVvFaceIACos;

    int polyFaceAmount;
    int polyLineAmout;

    int vvFaceAmout = 6;
    int vvLineAmout = 12;

    
};

#endif CAMERA_H_
