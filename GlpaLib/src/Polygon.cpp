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

Glpa::GPU_POLYGON Glpa::Polygon::getData(std::vector<Glpa::Vec3d*>& wv, std::vector<Glpa::Vec2d*>& uv)
{
    Glpa::GPU_POLYGON polygon;

    for (int i = 0; i < 3; i++)
    {
        polygon.wv[i].x = wv[wvI[i]]->x;
        polygon.wv[i].y = wv[wvI[i]]->y;
        polygon.wv[i].z = wv[wvI[i]]->z;

        polygon.uv[i].x = uv[uvI[i]]->x;
        polygon.uv[i].y = uv[uvI[i]]->y;
    }

    polygon.normal.x = normal.x;
    polygon.normal.y = normal.y;
    polygon.normal.z = normal.z;

    return polygon;
}
