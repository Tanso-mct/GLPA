#include "Scene3d.h"
#include "GlpaLog.h"

Glpa::Scene3d::Scene3d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    setType(3);
}

Glpa::Scene3d::~Scene3d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");

    for (auto& obj : objs)
    {
        delete obj.second;
    }

    for (auto& mt : mts)
    {
        delete mt.second;
    }
}

void Glpa::Scene3d::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_3D, "");
    for (auto& obj : objs)
    {
        obj.second->load();
    }

    for (auto& mt : mts)
    {
        mt.second->load();
    }
}

void Glpa::Scene3d::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_SCENE_3D, "");
    for (auto& obj : objs)
    {
        if (obj.second->isLoaded()) obj.second->release();
    }

    for (auto& mt : mts)
    {
        if (mt.second->isLoaded()) mt.second->release();
    }

    rend.dRelease();
}

void Glpa::Scene3d::AddMaterial(Glpa::Material *ptMt)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + ptMt->getName() + "]");
    ptMt->setManager(fileDataManager);
    mts.emplace(ptMt->getName(), ptMt);
}

void Glpa::Scene3d::DeleteMaterial(Glpa::Material *ptMt)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + ptMt->getName() + "]");
    ptMt->release();

    mts.erase(ptMt->getName());
    delete ptMt;
}

void Glpa::Scene3d::AddCamera(Glpa::Camera *ptCam)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Camera[" + ptCam->getName() + "]");
    cams.emplace(ptCam->getName(), ptCam);
}

void Glpa::Scene3d::SetCamera(Glpa::Camera *ptCam)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Camera[" + ptCam->getName() + "]");
    currentCam = ptCam->getName();
}

void Glpa::Scene3d::DeleteCamera(Glpa::Camera *ptCam)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Camera[" + ptCam->getName() + "]");
    cams.erase(ptCam->getName());
    delete ptCam;
}

void Glpa::Scene3d::rendering
(
    ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, HWND hWnd, PAINTSTRUCT ps,
    LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
){
    rend.run(objs, mts, *cams[currentCam], buf, bufWidth, bufHeight, bufDpi);

    D2D1_BITMAP_PROPERTIES bitmapProperties;
    bitmapProperties.pixelFormat = pRenderTarget->GetPixelFormat();
    bitmapProperties.dpiX = 96.0f;
    bitmapProperties.dpiY = 96.0f;

    D2D1_SIZE_U size = D2D1::SizeU(bufWidth, bufHeight);
    pRenderTarget->CreateBitmap(size, buf, bufWidth * sizeof(DWORD), &bitmapProperties, pBitMap);
}
