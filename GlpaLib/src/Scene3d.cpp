#include "Scene3d.h"

Glpa::Scene3d::Scene3d()
{
    setType(3);
}

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

void Glpa::Scene3d::rendering(ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    rend.run(objs, buf, bufWidth, bufHeight, bufDpi);

    D2D1_BITMAP_PROPERTIES bitmapProperties;
    bitmapProperties.pixelFormat = pRenderTarget->GetPixelFormat();
    bitmapProperties.dpiX = 96.0f;
    bitmapProperties.dpiY = 96.0f;

    D2D1_SIZE_U size = D2D1::SizeU(bufWidth, bufHeight);
    pRenderTarget->CreateBitmap(size, buf, bufWidth * sizeof(DWORD), &bitmapProperties, pBitMap);
}
