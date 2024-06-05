#include "Scene2d.h"

Glpa::Scene2d::~Scene2d()
{
}

void Glpa::Scene2d::load()
{
    for (auto& obj : objs)
    {
        obj.second->load();
    }
}

void Glpa::Scene2d::release()
{
    for (auto& obj : objs)
    {
        obj.second->release();
    }
}