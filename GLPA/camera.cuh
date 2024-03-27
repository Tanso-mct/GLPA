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
#include <algorithm>

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
    int poly_face_amount,
    int vv_line_amount,
    int vv_face_amount,
    int poly_line_amount
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
    int sort_4_vs_sizes,
    int sort_5_vs_sizes,
    int sort_6_vs_sizes,
    int sort_7_vs_sizes,
    double* sort_4_vs,
    double* sort_5_vs,
    double* sort_6_vs,
    double* sort_7_vs,
    double* vs_4_dot_cos,
    double* vs_4_cross,
    double* vs_5_dot_cos,
    double* vs_5_cross,
    double* vs_6_dot_cos,
    double* vs_6_cross,
    double* vs_7_dot_cos,
    double* vs_7_cross
);


__global__ void glpaGpuRasterize(
    double* left_side_sc_vs,
    double* right_side_sc_vs,
    double nearZ,
    double* poly_cam_one_vs,
    double* poly_cam_oneNs,
    int poly_amount,
    int* side_vs_size,
    int* sum_side_vs_size,
    int* side_per_size,
    int* rs_sum_size,
    double* near_screen_size,
    double* screen_pixel_size,
    double* rasterize_vs,
    double* rasterize_pixel_vs
);

/// @brief Has data related to the 3DCG camera.
class Camera{
public :
    void initialize();

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


    void setSortedVs(
        int faceAmout, int face_size, double* dot_cos, double* cross, std::vector<int> target_i, 
        std::vector<double> sortVs, std::vector<RasterizeSource>* pt_rasterize_source
    );

    void sortScPixelVs(std::vector<RasterizeSource>* pt_rasterize_source);
    

    void inputSideScRvs(
        double* left_side_screen_vs,
        double* right_side_screen_vs,
        int current_size,
        Vec2d pixel_v,
        int screen_pixel_vs_y_min
    );

    void rasterize(
        int v0_i,
        int v1_i,
        int rasterize_source_i,
        int screen_y_min,
        std::vector<RasterizeSource>* pt_rasterize_source,
        double* left_side_screen_vs,
        double* right_side_screen_vs,
        int current_size
    );

    void zBuffer(
        std::vector<RasterizeSource>* pt_rasterize_source, double* z_buffer_rasterize_source_i, double* z_buffer_vs
    );

    Matrix mt;
    Vector vec;

private : 
    bool reload = false;

    std::wstring name = GLPA_WSTRING_DEF;
    Vec3d wPos;
    Vec3d rotAngle;

    double nearZ;
    double farZ;
    double viewAngle;
    Vec2d viewAngleCos;
    Vec2d viewAngleTan;
    Vec2d aspectRatio;

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
    int polyLineAmount;

    int vvFaceAmount = 6;
    int vvLineAmount = 12;

};

#endif CAMERA_H_
