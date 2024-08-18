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

    polygon.wv0[X] = wv[wvI[0]]->x;
    polygon.wv0[Y] = wv[wvI[0]]->y;
    polygon.wv0[Z] = wv[wvI[0]]->z;

    polygon.wv1[X] = wv[wvI[1]]->x;
    polygon.wv1[Y] = wv[wvI[1]]->y;
    polygon.wv1[Z] = wv[wvI[1]]->z;

    polygon.wv2[X] = wv[wvI[2]]->x;
    polygon.wv2[Y] = wv[wvI[2]]->y;
    polygon.wv2[Z] = wv[wvI[2]]->z;

    polygon.uv0[X] = uv[uvI[0]]->x;
    polygon.uv0[Y] = uv[uvI[0]]->y;

    polygon.uv1[X] = uv[uvI[1]]->x;
    polygon.uv1[Y] = uv[uvI[1]]->y;

    polygon.uv2[X] = uv[uvI[2]]->x;
    polygon.uv2[Y] = uv[uvI[2]]->y;

    polygon.normal[X] = normal.x;
    polygon.normal[Y] = normal.y;
    polygon.normal[Z] = normal.z;

    return polygon;
}
