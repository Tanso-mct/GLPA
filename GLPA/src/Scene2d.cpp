#include "Scene2d.h"

Glpa::Scene2d::~Scene2d()
{
}

void Glpa::Scene2d::setDrawOrder()
{
    drawOrder.clear();
    imgAmount = 0;
    textAmount = 0;

    for (auto& obj : objs)
    {
        // TODO: Add processing with Text.
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj.second))
        {
            drawOrder[img->getDrawOrder()].push_back(img->getName());
            imgAmount++;
        }
    }
}

void Glpa::Scene2d::addDrawOrder(Glpa::SceneObject *obj)
{
    // TODO: Add processing with Text.
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        drawOrder[img->getDrawOrder()].push_back(img->getName());
        imgAmount++;
    }
}

void Glpa::Scene2d::deleteDrawOrder(Glpa::SceneObject *obj)
{
    // TODO: Add processing with Text.
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        std::vector<std::string>& order = drawOrder[img->getDrawOrder()];
        auto it = std::find(order.begin(), order.end(), obj->getName());
        if (it != order.end()) {
            order.erase(it);
            imgAmount--;
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
            addDrawOrder(obj.second);
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
            deleteDrawOrder(obj.second);
        }
    }
}

void Glpa::Scene2d::rendering(HDC dc, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    if (edited) rend.run(objs, drawOrder, dc, buf, bufWidth, bufHeight, bufDpi, backgroundColor);
}
