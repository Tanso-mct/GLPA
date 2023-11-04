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

class Camera
{
public :
    /**
     * @fn
     * To handle vector type variables, it is necessary to update the data array each time it is redrawn.
     * Therefore, initialize vector variables that need to be updated
     * Also initializes any variables that need initialization other than vector types
     * @brief Initialize variables that need to be initialized each time they are redrawn
     * @details
     * Initialize the small polygon information structure 
     * and the source, calculation, and search polygon information structures. 
     */
    void initialize();

    /**
     * @fn
     * Performs camera coordinate transformation for a rectangular area of a mesh, 
     * inputting and using data for a rectangular area from meshData.
     * @brief Perform camera coordinate transformation of the rectangular range of the mesh.
     * @param (mesh_data) The loaded 3D data is saved as a vector type variable for each mesh.
     * @details The 8 vertices of the rectangular range are obtained from the 3D coordinates of the origin and opposite 
     * vertex of the mesh at the read stage, and then transformed to camera coordinates, From those vertices, 
     * origin and opposite are determined again.
     */
    void coordinateTransRange(std::vector<OBJ_FILE>* mesh_data);

    // Determination of intersection of OBJECT with view volume
    void clipRange(std::vector<OBJ_FILE> objData);

    // Determining whether the face is front or back
    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> polyBilateralJudge(std::vector<OBJ_FILE> objData);

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTrans(std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>>, std::vector<OBJ_FILE> objData);

    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> objData);

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
