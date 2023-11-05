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
#include <windows.h>
#include <stdio.h>
#include <tuple>

#include "cgmath.cuh"
#include "file.h"
#include "cg.h"
#include "view_volume.h"

/// @brief Has data related to the 3DCG camera.
class Camera
{
public :
    /// @brief Initialize variables that need to be initialized each time they are redrawn.
    void initialize();

    /// @brief Perform camera coordinate transformation of the rectangular range of the mesh.
    /// @param mesh_data A summary of the imported 3D data stored together.
    void coordinateTransRange(std::vector<OBJ_FILE>* mesh_data);

    /// @brief Determine which meshes are in the view volume from the rectangular extent of the mesh.
    /// @param mesh_data A summary of the imported 3D data stored together.
    void clipRange(std::vector<OBJ_FILE> mesh_data);

    /// @brief Performs front-to-back determination of polygons in the view volume.
    /// @param mesh_data A summary of the imported 3D data stored together.
    /// @return World 3D vertex coordinates and world normal coordinates of the polygons to be transformed into camera
    /// coordinates.
    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> polyBilateralJudge(std::vector<OBJ_FILE> mesh_data);

    /// @brief Performs camera coordinate transformation of the vertices and normal vectors of the source polygon 
    /// information structure.
    /// @param mesh_data 
    void coordinateTrans(std::vector<OBJ_FILE> mesh_data);

    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> pushCalcPolyInfo(int loop_i);

    bool clipVerticesViewVolume();

    void inputRenderSourceSt();

    void inputCalcSourceSt();

    void createRangeFromPoly();

    void clipPolyRangeViewVolume();

    void clipPolyLineViewVolume();

    void judgeVertexInViewVolume();

    void clipViewVolumeLinePoly();

    void judgeVertexInPoly();

    // Determine if polygon is in view volume and store array number
    void clipPolyViewVolume(std::vector<OBJ_FILE> mesh_data);

    // Intersection judgment between polygon and view volume
    // std::vector<std::vector<int>> clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> range_polygon, int process_object_amout);

private : 
    VECTOR3D wPos = {0, 0, 0};
    VECTOR3D rotAngle = {0, 0, 0};

    double nearZ = 1;
    double farZ = 10000;
    ANGLE viewAngle = {0, 80};
    VECTOR2D aspectRatio = {16, 9};

    SIZE2 nearScrSize;
    SIZE2 farScrSize;

    ViewVolume viewVolume;
    
    SMALL_POLYINFO clipPolyInfo;
    std::vector<POLYINFO> sourcePolyInfo;
    std::vector<POLYINFO> calcPolyInfo;
    std::vector<POLYINFO> searchPolyInfo;

    std::vector<RENDERSOUCE> renderSouce;

    MATRIX mtx;
    VECTOR vec;
    EQUATION eq;
    TRIANGLE_RATIO tri;
};

#endif CAMERA_H_
