#include "GlpaBase.h"

GlpaBase::GlpaBase()
{
    awake();
}

GlpaBase::~GlpaBase()
{
    destroy();

    delete window;

    for (auto& sc : ptScs)
    {
        delete sc.second;
    }
}

void GlpaBase::AddScene(Glpa::Scene *ptScene)
{
    ptScene->setup();
    ptScs.emplace(ptScene->getName(), ptScene);
}

void GlpaBase::DeleteScene()
{
    ReleaseScene();

    std::string scName = ptScs[nowScName]->getName();

    delete ptScs[scName];
    ptScs.erase(scName);
}

void GlpaBase::DeleteScene(Glpa::Scene *ptScene)
{
    ReleaseScene(ptScene);

    std::string scName = ptScene->getName();

    delete ptScs[scName];
    ptScs.erase(scName);
}

void GlpaBase::DeleteAllScene()
{
    ReleaseAllScene();

    for (auto& sc : ptScs)
    {
        std::string scName = sc.second->getName();
        
        delete ptScs[scName];
    }

    ptScs.clear();
}

void GlpaBase::LoadScene()
{
    ptScs[nowScName]->load();
}

void GlpaBase::LoadScene(Glpa::Scene *ptScene)
{
    nowScName = ptScene->getName();
    ptScene->load();
}

void GlpaBase::ReleaseScene()
{
    ptScs[nowScName]->release();
}

void GlpaBase::ReleaseScene(Glpa::Scene *ptScene)
{
    ptScene->release();
}

void GlpaBase::ReleaseAllScene()
{
    for (auto& sc : ptScs)
    {
        sc.second->release();
    }
}

void GlpaBase::SetFirstSc(Glpa::Scene *ptScene)
{
    startScName = ptScene->getName();
}

void GlpaBase::runStart()
{
    start();
    ptScs[nowScName]->start();
    started = true;

    window->sendPaintMsg();
}

void GlpaBase::runUpdate()
{
    update();
    ptScs[nowScName]->update();

    window->sendPaintMsg();
}
