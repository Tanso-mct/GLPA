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

            rend.dRelease();
        }
    }
}

void Glpa::Scene2d::rendering(ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    rend.run(objs, drawOrder, buf, bufWidth, bufHeight, bufDpi, backgroundColor);

    D2D1_BITMAP_PROPERTIES bitmapProperties;
    bitmapProperties.pixelFormat = pRenderTarget->GetPixelFormat();
    // bitmapProperties.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bitmapProperties.dpiX = 96.0f * bufDpi;
    bitmapProperties.dpiY = 96.0f * bufDpi;

    D2D1_SIZE_U size = D2D1::SizeU(bufWidth, bufHeight);
    HRESULT hr = pRenderTarget->CreateBitmap(size, buf, bufWidth * sizeof(DWORD), &bitmapProperties, pBitMap);
}
