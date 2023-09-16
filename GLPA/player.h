#ifndef PLAYER_H_
#define PLAYER_H_

#include "file.h"
#include "cgmath.cuh"

class PLAYER : public OBJ_FILE
{
public :
    //TODO: Creating initialization functions
    VECTOR3D wPos;
    VECTOR3D rotAngle;
    VECTOR3D scaleRate;

    SIZE2 hitBoxSize;
};



#endif PLAYER_H_
