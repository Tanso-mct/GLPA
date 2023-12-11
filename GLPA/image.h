#ifndef IMAGE_H_
#define IMAGE_H_

#include "vector.cuh"
#include "png.h"

class Image
{
public :
    void move();
    bool judegeMouseOn();

    bool visible = true;
    Png png;

private :
    Vec2d pos;
};


#endif  IMAGE_H_