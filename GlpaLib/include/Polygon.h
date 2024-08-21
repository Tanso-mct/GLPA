#ifndef GLPA_POLYGON_H_
#define GLPA_POLYGON_H_

#include <vector>
#include <string>

#include "Vector.cuh"


namespace Glpa
{

typedef struct _GPU_POLYGON
{
    Glpa::GPU_VEC_3D wv[3];
    Glpa::GPU_VEC_2D uv[3];
    Glpa::GPU_VEC_3D normal;
} GPU_POLYGON;

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

    Glpa::GPU_POLYGON getData(std::vector<Glpa::Vec3d*>& wv, std::vector<Glpa::Vec2d*>& uv);
};

}


#endif GLPA_POLYGON_H_