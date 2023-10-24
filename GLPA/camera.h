#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <windows.h>
#include <stdio.h>

#include "cgmath.cuh"
#include "file.h"
#include "graphic.h"

#define VP1 0
#define VP2 1
#define VP3 2
#define VP4 3

class CAMERA
{
public :
    VECTOR3D wPos;
    VECTOR3D rotAngle;

    bool initialized = false;

    double nearZ;
    double farZ;
    double viewAngle;
    VECTOR2D aspectRatio;

    SIZE2 nearScreenSize;
    SIZE2 farScreenSize;
    
    std::vector<VECTOR3D> viewPoint;
    std::vector<VECTOR_XZ> viewPointXZ;
    std::vector<VECTOR_YZ> viewPointYZ;

    std::vector<VECTOR3D> viewVolumeFaceVertex;
    std::vector<VECTOR3D> viewVolumeFaceNormal;

    MATRIX mtx;
    VECTOR vec;
    EQUATION eq;

    std::vector<int> withinRangeAryNum;
    std::vector<std::vector<int>> numPolyFacing;

    std::vector<VECTOR3D> polyVertex;
    std::vector<VECTOR3D> polyNormal;

    std::vector<std::vector<int>> numPolyInViewVolume;

    // Polygon clipped vertices in view volume
    // The index is numPolyInViewVolume
    std::vector<std::vector<std::vector<VECTOR3D>>> clippedPolyVertex;

    // Stores polygons where possible intersections between polygon line segments and view volume surfaces exist
    std::vector<std::vector<int>> numPolyExitsIViewVolume;

    // Stores polygons with intersections with the view volume plane
    std::vector<std::vector<int>> numPolyTrueIViewVolume;

    // Store polygons that are outside of the view volume at all three points and all three sides
    std::vector<std::vector<int>> numPolyAllVLINENotInViewVolume;

    void initialize(); // Initialize data
    void defViewVolume(); // define clipping area

    // Range coordinate transformation
    void coordinateTransRange(std::vector<OBJ_FILE>* objData);

    // Determination of intersection of OBJECT with view volume
    void clippingRange(std::vector<OBJ_FILE> objData);

    // Determining whether the face is front or back
    void polyBilateralJudge(std::vector<OBJ_FILE> objData);

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTrans(std::vector<OBJ_FILE> objData);

    // Determines if a vertex is in the view volume
    bool vertexInViewVolume(VECTOR3D vertex);
    
    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> objData);

    // Intersection judgment between polygon and view volume
    std::vector<std::vector<int>> clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> range_polygon, int process_object_amout);

};

#endif CAMERA_H_
