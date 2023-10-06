#ifndef HITBOX_CUH_
#define HITBOX_CUH_

#include "cgmath.cuh"

class HITBOX
{
public :
    SIZE2 boxSize;
    RANGE_CUBE boxRange;
    void examineHit();
};

#endif HITBOX_CUH_
