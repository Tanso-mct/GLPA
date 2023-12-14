#ifndef IMAGE_H_
#define IMAGE_H_

#include "vector.cuh"
#include "png.h"

class Image
{
public :
    void show();
    void move();
    void scale();
    void rotate();
    bool judgeMouseOn();

    bool visible = true;
    Png png;
    Vec2d pos;
};


#endif  IMAGE_H_