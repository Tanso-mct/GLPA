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
    /// @param mesh_data A summary of the imported 3D data stored together.
    void coordinateTrans(std::vector<OBJ_FILE> mesh_data);

    /// @brief The polygon information structure to be calculated and the render source structure are created. 
    /// Polygon line vectors are not calculated by this function.
    /// @param loop_i Loop counter variable in the source polygon information structure.
    /// @return 3D vectors of line segment viewpoints and endpoints used to create 3D vectors of polygonal line segments.
    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> pushCalcPolyInfo(int loop_i);

    /// @brief Using the loop counter variable looped in the source polygon information structure, the polygon 
    /// information structure to be calculated and the rendering source structure are created from the data 
    /// corresponding to the variable.
    /// @param loop_i Loop counter variable in the source polygon information structure.
    /// @param pt_calc_poly_start_line A pointer to something that collectively stores the 3D coordinates of the 
    /// starting point of a polygon line segment.
    /// @param pt_calc_poly_end_line A pointer to a variable that collectively stores the 3D coordinates of the 
    /// starting point of a polygon line segment.
    /// @param pt_alredy_pushed A pointer to a variable to be specified to indicate that it has been done, since this 
    /// process is done only once for each polygon.
    void createStRenderSourceCalcPolyInfo
    (
        int loop_i,
        std::vector<VECTOR3D>* pt_calc_poly_start_line, std::vector<VECTOR3D>* pt_calc_poly_end_line,
        bool* pt_alredy_pushed
    );

    /// @brief Using variables that summarize the 3D coordinates of the start and end points of the polygon line 
    /// segment in the polygon information structure to be calculated, the 3D vector of the polygon line segment is 
    /// obtained.
    /// @param line_start_point 3-D coordinates of the starting point of a polygon line segment stored together.
    /// @param line_end_point 3-D coordinates of the endpoints of polygonal line segments stored together.
    void calcPolyLineVec(std::vector<VECTOR3D> line_start_point, std::vector<VECTOR3D> line_end_point);

    /// @brief The vertex data in the source polygon information structure is clipped in the view volume, and the result 
    /// inputs information to the target polygon information structure for calculation and retrieval and the rendering 
    /// source structure. 
    /// @include pushCalcPolyInfo()
    /// @include createStRenderSourceCalcPolyInfo()
    /// @include calcPolyLineVec()
    void clipVerticesViewVolume();

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
