#ifndef CAMERA_H_
#define CAMERA_H_

#include "cgmath.cuh"

class CAMERA
{
public :
    VECTOR3D wPos;
    VECTOR3D rotAngle;

    double nearZ;
    double farZ;
    double viewAngle;
    VECTOR2D aspectRatio;
    SIZE2 screenSize;

    void initialize(); // Initialize data
    void defClippingArea(); // define clipping area
    void clippingRange(); // Sellect range by clipping area
    void polyBilateralJudge(); // Determining whether the face is front or back

    // Coordinate transformation of the vertices of the surface to be drawn
    void coordinateTrans();
};

#endif CAMERA_H_
