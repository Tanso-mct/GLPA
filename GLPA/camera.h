#ifndef CAMERA_H_
#define CAMERA_H_

#include <vector>
#include <math.h>
#include <windows.h>
#include <stdio.h>

#include "cgmath.cuh"
#include "file.h"
#include "graphic.h"\

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
    std::vector<INT2D> numPolyFacing;
    std::vector<VECTOR3D> polyVertex;
    std::vector<INT2D> numPolyInViewVolume;

    void initialize(); // Initialize data
    void defViewVolume(); // define clipping area

    // Range coordinate transformation
    void coordinateTransRange(std::vector<OBJ_FILE>* objData);

    void clippingRange(std::vector<OBJ_FILE> objData);

    // Determining whether the face is front or back
    void polyBilateralJudge(std::vector<OBJ_FILE> objData);

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTransV(std::vector<OBJ_FILE> objData);

    bool confirmI
    (
        line_I_amout_data,
        
    )
    // Determine if polygon is in view volume and store array number
    void polyInViewVolumeJudge(std::vector<OBJ_FILE> objData);
};

#endif CAMERA_H_
