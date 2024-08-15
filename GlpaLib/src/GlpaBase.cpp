#include "GlpaBase.h"
#include "GlpaLog.h"

GlpaBase::GlpaBase()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    window = new Glpa::Window();
    
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Base[" + name + "] awake()");
    awake();
}

GlpaBase::~GlpaBase()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Base[" + name + "] destroy()");
    destroy();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete window");
    delete window;

    for (auto& sc : ptScs)
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete scene[" + sc.first + "]");
        delete sc.second;
    }
}

void GlpaBase::AddScene(Glpa::Scene *ptScene)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Scene[" + ptScene->getName() + "]");
    ptScene->setManager(fileDataManager);
    ptScene->setWindow(window);

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Scene[" + ptScene->getName() + "] setup()");
    ptScene->setup();

    ptScs.emplace(ptScene->getName(), ptScene);
}

void GlpaBase::DeleteScene()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + nowScName + "]");
    ReleaseScene();

    std::string scName = ptScs[nowScName]->getName();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete scene[" + scName + "]");
    delete ptScs[scName];
    ptScs.erase(scName);
}

void GlpaBase::DeleteScene(Glpa::Scene *ptScene)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + ptScene->getName() + "]");
    ReleaseScene(ptScene);

    std::string scName = ptScene->getName();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete scene[" + scName + "]");
    delete ptScs[scName];
    ptScs.erase(scName);
}

void GlpaBase::DeleteAllScene()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "All scene");
    ReleaseAllScene();

    for (auto& sc : ptScs)
    {
        std::string scName = sc.second->getName();
        
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete scene[" + scName + "]");
        delete ptScs[scName];
    }

    ptScs.clear();
}

void GlpaBase::LoadScene()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + nowScName + "]");
    if (!getStarted())
    {
        nowScName = startScName;
    }

    loadingSceneCount++;
    ptScs[nowScName]->load();
}

void GlpaBase::LoadScene(Glpa::Scene *ptScene)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + ptScene->getName() + "]");
    nowScName = ptScene->getName();

    loadingSceneCount++;
    ptScene->load();
}

void GlpaBase::ReleaseScene()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + nowScName + "]");

    loadingSceneCount--;
    ptScs[nowScName]->release();
}

void GlpaBase::ReleaseScene(Glpa::Scene *ptScene)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + ptScene->getName() + "]");

    loadingSceneCount--;
    ptScene->release();
}

void GlpaBase::ReleaseAllScene()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "All scene");

    for (auto& sc : ptScs)
    {
        sc.second->release();
    }

    loadingSceneCount = 0;
}

void GlpaBase::SetFirstSc(Glpa::Scene *ptScene)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_BASE, "Scene[" + ptScene->getName() + "]");
    startScName = ptScene->getName();
}

void GlpaBase::runStart()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Base[" + name + "] start()");
    start();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Scene[" + nowScName + "] start()");
    ptScs[nowScName]->start();

    started = true;

    if (window->pBitmap != nullptr) window->pBitmap->Release();

    ptScs[nowScName]->rendering
    (
        window->pRenderTarget, &window->pBitmap, window->hWnd, window->hPs,
        window->pixels, window->GetWidth(), window->GetHeight(), window->GetDpi()
    );

    ptScs[nowScName]->updateKeyMsg();
    ptScs[nowScName]->updateMouseMsg();
}

void GlpaBase::runUpdate()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB_FRAME, "Base[" + name + "] update()");
    update();

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB_FRAME, "Scene[" + nowScName + "] update()");
    ptScs[nowScName]->update();

    if (window->pBitmap != nullptr) window->pBitmap->Release();

    ptScs[nowScName]->rendering
    (
        window->pRenderTarget, &window->pBitmap, window->hWnd, window->hPs,
        window->pixels, window->GetWidth(), window->GetHeight(), window->GetDpi()
    );

    ptScs[nowScName]->updateKeyMsg();
    ptScs[nowScName]->updateMouseMsg();
}
