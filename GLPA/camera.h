#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <windows.h>
#include <stdio.h>

#include "cgmath.cuh"
#include "file.h"
#include "cg.h"
#include "view_volume.h"

class Camera
{
public :
    Camera()
    {
        
    }

    // Range coordinate transformation
    void coordinateTransRange(std::vector<OBJ_FILE>* objData);

    // Determination of intersection of OBJECT with view volume
    void clipRange(std::vector<OBJ_FILE> objData);

    // Determining whether the face is front or back
    void polyBilateralJudge(std::vector<OBJ_FILE> objData);

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTrans(std::vector<OBJ_FILE> objData);

    // Determines if a vertex is in the view volume
    std::vector<bool> vertexInViewVolume(std::vector<VECTOR3D> vertex);
    
    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> objData);

    // Intersection judgment between polygon and view volume
    std::vector<std::vector<int>> clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> range_polygon, int process_object_amout);

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
    
    std::vector<MESHINFO> meshInfo;

    std::vector<std::vector<SMALL_POLYINFO>> clipPolyInfo;
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
