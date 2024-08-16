#include "Scene2d.h"
#include "GlpaLog.h"

Glpa::Scene2d::Scene2d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    setType(2);
}

Glpa::Scene2d::~Scene2d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Scene2d::editPos(Glpa::Image *img, Glpa::Vec2d newPos)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + img->getName() + "]");
    img->SetPos(newPos);

    rend.editObjsPos(img);
}

void Glpa::Scene2d::EditDrawOrder(Glpa::SceneObject *obj, int newDrawOrder)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Object[" + obj->getName() + "]");
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + img->getName() + "]");
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Object[" + obj->getName() + "]");
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + img->getName() + "]");
        drawOrderMap[img->GetDrawOrder()].push_back(img->getName());
        imgAmount++;
    }
}

void Glpa::Scene2d::deleteDrawOrder(Glpa::SceneObject *obj)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Object[" + obj->getName() + "]");
    if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(obj))
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + img->getName() + "]");
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "");
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "");
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "");
    if (drawOrder.empty()) return "";

    for (int i = drawOrder.size() - 1; i >= 0; i--)
    {
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[drawOrder[i]]))
        {
            Glpa::Vec2d imgPos = img->GetPos();
            int imgWidth = img->GetWidth();
            int imgHeight = img->GetHeight();

            if (pos.x >= imgPos.x && pos.y >= imgPos.y && pos.x < imgPos.x + imgWidth && pos.y < imgPos.y + imgHeight)
            {
                Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + img->getName() + "]");
                return img->getName();
            }
        }
    }
    
    return "";
}

bool Glpa::Scene2d::GetIsImageAtPos(Glpa::Vec2d pos, std::string imgName)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "Image[" + imgName + "]");
    if (drawOrder.empty()) return "";

    for (int i = drawOrder.size() - 1; i >= 0; i--)
    {
        if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[drawOrder[i]]))
        {
            Glpa::Vec2d imgPos = img->GetPos();
            int imgWidth = img->GetWidth();
            int imgHeight = img->GetHeight();

            if (pos.x >= imgPos.x && pos.y >= imgPos.y && pos.x < imgPos.x + imgWidth && pos.y < imgPos.y + imgHeight)
            {
                if (img->getName() == imgName)
                {
                    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_2D, "True");
                    return true;
                }
            }
        }
    }
    
    return false;
}

void Glpa::Scene2d::rendering
(
    ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, HWND hWnd, PAINTSTRUCT ps,
    LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
)
{
    rend.run(objs, drawOrderMap, drawOrder, buf, bufWidth, bufHeight, bufDpi, backgroundColor);

    D2D1_BITMAP_PROPERTIES bitmapProperties;
    bitmapProperties.pixelFormat = pRenderTarget->GetPixelFormat();
    bitmapProperties.dpiX = 96.0f * bufDpi;
    bitmapProperties.dpiY = 96.0f * bufDpi;

    D2D1_SIZE_U size = D2D1::SizeU(bufWidth, bufHeight);
    HRESULT hr = pRenderTarget->CreateBitmap(size, buf, bufWidth * sizeof(DWORD), &bitmapProperties, pBitMap);
    
    ID2D1Bitmap* bitMap = *pBitMap;

    BeginPaint(hWnd, &ps);
    pRenderTarget->BeginDraw();
    
    D2D1_SIZE_F bitMapSize = bitMap->GetSize();
    D2D1_RECT_F layoutRect = D2D1::RectF(0, 0, bitMapSize.width, bitMapSize.height);

    pRenderTarget->DrawBitmap(bitMap, layoutRect);

    //TODO: Add Text drawing processing here.
    for (auto& obj : objs)
    {
        if (Glpa::Text* text = dynamic_cast<Glpa::Text*>(obj.second))
        {
            text->drawText(pRenderTarget);
        }
    }

    
    pRenderTarget->EndDraw();
    EndPaint(hWnd, &ps);
}
