#include "Scene2d.h"

Glpa::Scene2d::~Scene2d()
{
}

void Glpa::Scene2d::setDrawOrder()
{
    drawOrder.clear();

    for (auto& obj : objs)
    {
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj.second))
        {
            drawOrder[img->getDrawOrder()].push_back(img->getName());
        }
    }
}

void Glpa::Scene2d::addDrawOrder(Glpa::SceneObject *obj)
{
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        drawOrder[img->getDrawOrder()].push_back(img->getName());
    }
}

void Glpa::Scene2d::deleteDrawOrder(Glpa::SceneObject *obj)
{
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        drawOrder[img->getDrawOrder()].erase
        (
            drawOrder[img->getDrawOrder()].begin() + std::distance
            (
                drawOrder[img->getDrawOrder()].begin(),
                std::find(drawOrder[img->getDrawOrder()].begin(), drawOrder[img->getDrawOrder()].end(), obj)
            )
        )
    }
}
