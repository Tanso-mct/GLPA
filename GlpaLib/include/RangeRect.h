#ifndef GLPA_RANGE_RECT_H_
#define GLPA_RANGE_RECT_H_

#include <vector>

#include "Vector.h"

namespace Glpa
{

class RangeRect
{
private :
    bool isStatus = false;
    Glpa::Vec3d origin;
    Glpa::Vec3d opposite;
    std::vector<Glpa::Vec3d> wvs;

};


}

#endif GLPA_RANGE_RECT_H_