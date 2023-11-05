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

    /**
     * @fn
     * Determine which meshes are in the view volume from the rectangular extent of the mesh.
     * @brief Determine which meshes are in the view volume from the rectangular extent of the mesh.
     * @param (mesh_data) The loaded 3D data is saved as a vector type variable for each mesh.
     * @details Calculate the angles of the vertices in the rectangular range of the mesh data in the XZ and YZ axes, 
     * determine whether the mesh is in the view volume based on the angles, and store the data of the mesh in the view 
     * volume in the small polygon information structure.
     */
    void clipRange(std::vector<OBJ_FILE> mesh_data);

    // Determining whether the face is front or back
    /**
     * @fn
     * The vertices and normal 3D vectors from the data of the small polygon information structure are converted to 
     * camera coordinates to determine the front and back of polygons.
     * @brief Performs front-to-back determination of polygons in the view volume.
     * @param (mesh_data) The loaded 3D data is saved as a vector type variable for each mesh.
     * @return A variable of type tuple that stores the three vertices of the polygons in the view volume that face the
     * table and the normal vector of the face.
     * @details The world 3D vertex coordinates and world polygon normal vector in the small polygon information
     * structure are converted to camera coordinates. In this case, the normal vectors are normalized, so the position
     * information remains unchanged. Using these camera 3D vertex coordinates and camera polygon normal vectors, the
     * camera determines whether the surface is facing the front or not. If the result is facing the surface, the three
     * vertices of the polygon and the normal vector are stored in a vector type variable and returned as a return
     * value using tuple. This value is used in the next camera coordinate transformation function.
     */
    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> polyBilateralJudge(std::vector<OBJ_FILE> mesh_data);

    // Coordinate transformation of the vertices of the surface to be drawn
    /**
     * @fn
     * The world 3D vertex coordinates of the source polygon information structure and the camera coordinate
     * transformation of the world normal vector are performed using the return value of the polyBilateralJudge 
     * function.
     * @brief Performs camera coordinate transformation of the vertices and normal vectors of the source polygon 
     * information structure.
     * @param (mesh_data) The loaded 3D data is saved as a vector type variable for each mesh.
     * @sa polyBilateralJudge()
     * @details Since the order of the array of vector variables in the return value of the polyBilateralJudge function 
     * and the order of the array of the source polygon information structure are equal, the return value is camera 
     * coordinate transformed and the camera coordinate transformed data is input into the source polygon information 
     * structure in turn by loop processing.
     */
    void coordinateTrans(std::vector<OBJ_FILE> mesh_data);

    bool clipVerticesViewVolume();

    void inputRenderSouceSt();

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
