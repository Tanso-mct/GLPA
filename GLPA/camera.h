#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <windows.h>
#include <stdio.h>

#include "cgmath.cuh"
#include "file.h"

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
    
    std::vector<VECTOR_XZ> viewPointXZ;
    std::vector<VECTOR_YZ> viewPointYZ;

    MATRIX mtx;

    void initialize(); // Initialize data
    void defClippingArea(); // define clipping area

    // Range coordinate transformation
    void coordinateTransRange
    (
        std::vector<OBJ_FILE>& objData
    );

    void clippingRange(); // Sellect range by clipping area
    void polyBilateralJudge(); // Determining whether the face is front or back

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTransV();
};

extern CAMERA mainCam;

#endif CAMERA_H_
