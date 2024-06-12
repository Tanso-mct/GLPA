#include "Scene3d.h"

Glpa::Scene3d::~Scene3d()
{
}

void Glpa::Scene3d::load()
{
    for (auto& obj : objs)
    {
        obj.second->load();
    }
}

void Glpa::Scene3d::release()
{
    for (auto& obj : objs)
    {
        if (obj.second->isLoaded()) obj.second->release();
    }
}

void Glpa::Scene3d::rendering(HDC dc, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    rend.run(objs, dc, buf);
}
