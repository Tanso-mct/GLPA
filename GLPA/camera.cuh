/**
 * @file Camera.h
 * @brief Describes a process related to a camera that exists in 3D
 * @author Tanso
 * @date 2023-10
 */

#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <string>

#include "cg.h"
#include "view_volume.cuh"



/// @brief Has data related to the 3DCG camera.
class Camera{
public :
    void load(
        std::wstring argName,
        Vec3d argWPos,
        Vec3d argRotAngle,
        double argNearZ,
        double argFarZ,
        Vec2d argViewAngle,
        Vec2d argAspectRatio
    );

    void defineViewVolume();

    void objRangeCoordTrans();

    void objCulling();

    void meshCulling();

    void polyBilateralJudge();

    void polyCulling();

    void polyShapeConvert();


private : 
    std::wstring name = GLPA_WSTRING_DEF;
    Vec3d wPos = {0, 0, 0};
    Vec3d rotAngle = {0, 0, 0};

    double nearZ = 1;
    double farZ = 10000;
    Vec2d viewAngle = {0, 80};
    Vec2d aspectRatio = {16, 9};

    Vec2d nearScrSize;
    Vec2d farScrSize;

    ViewVolume viewVolume;
    
    std::vector<std::wstring> renderTargetObj;
    std::vector<MeshNameInfo> renderTargetMesh;
    std::vector<PolyNameInfo> renderTargetPoly;
    std::vector<PolyNameInfo> shapeConvertTargetPoly;

    
};

#endif CAMERA_H_
