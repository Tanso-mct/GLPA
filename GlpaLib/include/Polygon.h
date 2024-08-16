#ifndef GLPA_POLYGON_H_
#define GLPA_POLYGON_H_

#include <vector>
#include <string>

#include "Vector.h"


namespace Glpa
{

class Polygon
{
private :
    std::vector<int> wvI;
    std::vector<int> uvI;
    Glpa::Vec3d normal;

    std::string mtName;

public :
    Polygon();
    ~Polygon();

    void addWv(Glpa::Vec3d argWv);
    void addUv(Glpa::Vec2d argUv);
    void addNormal(Glpa::Vec3d argNormal);

    Glpa::Vec3d getWv(int index);
    Glpa::Vec2d getUv(int index);
    Glpa::Vec3d getNormal(int index);
};

}


#endif GLPA_POLYGON_H_