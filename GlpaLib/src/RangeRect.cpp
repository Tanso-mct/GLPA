#include "RangeRect.h"
#include "GlpaLog.h"

Glpa::RangeRect::RangeRect()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::RangeRect::~RangeRect()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::RangeRect::addRangeV(Glpa::Vec3d *wv)
{
    if (!isStatus)
    {
        origin = (*wv);
        opposite = (*wv);
    }
    else
    {
        if ((*wv).x < origin.x) origin.x = (*wv).x;
        if ((*wv).y < origin.y) origin.y = (*wv).y;
        if ((*wv).z > origin.z) origin.z = (*wv).z;

        if ((*wv).x > opposite.x) opposite.x = (*wv).x;
        if ((*wv).y > opposite.y) opposite.y = (*wv).y;
        if ((*wv).z < opposite.z) opposite.z = (*wv).z;
    }

    isStatus = true;
}

void Glpa::RangeRect::setWvs()
{
    wvs.clear();
    wvs.push_back(Glpa::Vec3d(origin.x, opposite.y, origin.z));
    wvs.push_back(Glpa::Vec3d(opposite.x, opposite.y, origin.z));
    wvs.push_back(Glpa::Vec3d(opposite.x, origin.y, origin.z));
    wvs.push_back(Glpa::Vec3d(origin.x, origin.y, origin.z));
    wvs.push_back(Glpa::Vec3d(origin.x, opposite.y, opposite.z));
    wvs.push_back(Glpa::Vec3d(opposite.x, opposite.y, opposite.z));
    wvs.push_back(Glpa::Vec3d(opposite.x, origin.y, opposite.z));
    wvs.push_back(Glpa::Vec3d(origin.x, origin.y, opposite.z));
}
