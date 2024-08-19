#include "Polygon.h"
#include "GlpaLog.h"

Glpa::Polygon::Polygon()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::Polygon::~Polygon()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Polygon::addV(int argWvI, int argUvI)
{
    wvI.push_back(argWvI);
    uvI.push_back(argUvI);
}

Glpa::POLYGON Glpa::Polygon::getData(std::vector<Glpa::Vec3d*>& wv, std::vector<Glpa::Vec2d*>& uv)
{
    Glpa::POLYGON polygon;

    for (int i = 0; i < 3; i++)
    {
        polygon.wv[i][X] = wv[wvI[i]]->x;
        polygon.wv[i][Y] = wv[wvI[i]]->y;
        polygon.wv[i][Z] = wv[wvI[i]]->z;

        polygon.uv[i][X] = uv[uvI[i]]->x;
        polygon.uv[i][Y] = uv[uvI[i]]->y;
    }

    polygon.normal[X] = normal.x;
    polygon.normal[Y] = normal.y;
    polygon.normal[Z] = normal.z;

    return polygon;
}
