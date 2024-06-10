#include "Scene2d.h"

Glpa::Scene2d::~Scene2d()
{
}

void Glpa::Scene2d::setDrawOrder()
{
    for (auto& obj : objs)
    {
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj.second))
        {
            
        }
    }
}
