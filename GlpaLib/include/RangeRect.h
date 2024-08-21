#ifndef GLPA_RANGE_RECT_H_
#define GLPA_RANGE_RECT_H_

#include <vector>
#include <Windows.h>

#include "Vector.h"

namespace Glpa
{

typedef struct _RANGE_RECT
{
    GPU_VEC_3D origin;
    GPU_VEC_3D opposite;

    GPU_VEC_3D wv[8];
} RANGE_RECT;

typedef struct _GPU_RANGE_RECT
{
    GPU_BOOL isEmpty = FALSE;

    GPU_VEC_3D origin;
    GPU_VEC_3D opposite;
} GPU_RANGE_RECT;

class RangeRect
{
private :
    bool isStatus = false;
    Glpa::Vec3d origin;
    Glpa::Vec3d opposite;
    std::vector<Glpa::Vec3d> wvs;

public :
    RangeRect();
    ~RangeRect();

    Glpa::Vec3d getOrigin() const {return origin;}
    Glpa::Vec3d getOpposite() const {return opposite;}

    void addRangeV(Glpa::Vec3d* wv);

    void setWvs();
    Glpa::RANGE_RECT getData();

    void setStatus(bool value) {isStatus = value;}
    bool getStatus() const {return isStatus;}

};


}

#endif GLPA_RANGE_RECT_H_