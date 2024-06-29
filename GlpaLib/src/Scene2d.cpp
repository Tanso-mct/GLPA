#include "Scene2d.h"

Glpa::Scene2d::Scene2d()
{
    setType(2);
}

Glpa::Scene2d::~Scene2d()
{
}

void Glpa::Scene2d::editPos(Glpa::Image *img, Glpa::Vec2d newPos)
{
    img->SetPos(newPos);

    rend.editObjsPos(img);
}

void Glpa::Scene2d::EditDrawOrder(Glpa::SceneObject *obj, int newDrawOrder)
{
    // TODO: Add processing with Text.
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        drawOrderMap[img->GetDrawOrder()].push_back(img->getName());

        drawOrderMap[img->GetDrawOrder()].erase
        (
            std::find(drawOrderMap[img->GetDrawOrder()].begin(), drawOrderMap[img->GetDrawOrder()].end(), img->getName())
        );

        img->SetDrawOrder(newDrawOrder);

        addDrawOrder(obj);
    }
}

void Glpa::Scene2d::addDrawOrder(Glpa::SceneObject *obj)
{
    // TODO: Add processing with Text.
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        drawOrderMap[img->GetDrawOrder()].push_back(img->getName());
        imgAmount++;
    }
}

void Glpa::Scene2d::deleteDrawOrder(Glpa::SceneObject *obj)
{
    // TODO: Add processing with Text.
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        std::vector<std::string>& order = drawOrderMap[img->GetDrawOrder()];
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

std::string Glpa::Scene2d::GetNowImageAtPos(Glpa::Vec2d pos)
{
    if (drawOrder.empty()) return "";

    for (int i = drawOrder.size() - 1; i >= 0; i--)
    {
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[drawOrder[i]]))
        {
            Glpa::Vec2d imgPos = img->GetPos();
            int imgWidth = img->getWidth();
            int imgHeight = img->getHeight();

            if (pos.x >= imgPos.x && pos.y >= imgPos.y && pos.x < imgPos.x + imgWidth && pos.y < imgPos.y + imgHeight)
            {
                return img->getName();
            }
        }
    }
    
    return "";
}

void Glpa::Scene2d::rendering(ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    rend.run(objs, drawOrderMap, drawOrder, buf, bufWidth, bufHeight, bufDpi, backgroundColor);

    D2D1_BITMAP_PROPERTIES bitmapProperties;
    bitmapProperties.pixelFormat = pRenderTarget->GetPixelFormat();
    bitmapProperties.dpiX = 96.0f * bufDpi;
    bitmapProperties.dpiY = 96.0f * bufDpi;

    D2D1_SIZE_U size = D2D1::SizeU(bufWidth, bufHeight);
    HRESULT hr = pRenderTarget->CreateBitmap(size, buf, bufWidth * sizeof(DWORD), &bitmapProperties, pBitMap);
}
