#ifndef PLAYER_H_
#define PLAYER_H_

#include "file.h"
#include "cgmath.cuh"

#include "camera.h"
#include "hitbox.cuh"

class PLAYER
{
public :
    //TODO: Creating initialization functions
    VECTOR3D wPos;
    VECTOR3D rotAngle;
    VECTOR3D scaleRate;

    // Player object data
    OBJ_FILE body;
    OBJ_FILE head;

    SIZE2 hitBoxSize;

    CAMERA cam;


};



#endif PLAYER_H_
