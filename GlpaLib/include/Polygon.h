#ifndef GLPA_POLYGON_H_
#define GLPA_POLYGON_H_

#include <vector>
#include <string>

#include "Vector.h"


namespace Glpa
{

typedef struct _POLYGON
{
    float wv[3][3];
    float uv[3][2];
    float normal[3];
} POLYGON;

class Polygon
{
private :
    std::vector<int> wvI;
    std::vector<int> uvI;
    Glpa::Vec3d normal;

public :
    Polygon();
    ~Polygon();

    void addV(int argWvI, int argUvI);
    void setNormal(Glpa::Vec3d argNormal){normal = argNormal;};

    Glpa::POLYGON getData(std::vector<Glpa::Vec3d*>& wv, std::vector<Glpa::Vec2d*>& uv);
};

}


#endif GLPA_POLYGON_H_