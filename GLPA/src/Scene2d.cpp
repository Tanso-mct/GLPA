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
        auto& order = drawOrder[img->getDrawOrder()];
        auto it = std::find(order.begin(), order.end(), obj->getName());
        if (it != order.end()) {
            order.erase(it);
        }
    }
}

void Glpa::Scene2d::load()
{
    for (auto& obj : objs)
    {
        if (!obj.second->isLoaded()) 
        {
            obj.second->load();
            // addDrawOrder(obj.second);
        }
    }
}

void Glpa::Scene2d::release()
{
    for (auto& obj : objs)
    {
        if (obj.second->isLoaded()) 
        {
            obj.second->release();
            // deleteDrawOrder(obj.second);
        }
    }
}
