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
    SIZE2 screenSize;
};

#endif CAMERA_H_
