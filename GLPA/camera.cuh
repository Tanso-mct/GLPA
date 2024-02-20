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

    void polyCulling();

    void polyShapeConvert();

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
    std::vector<PolyNameInfo> shapeConvertTargetPoly;


    
};

#endif CAMERA_H_
