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

void GlpaBase::DeleteScene(Glpa::Scene *ptScene)
{
    delete ptScs[ptScene->getName()];
    ptScs.erase(ptScene->getName());
}

void GlpaBase::loadScene()
{
    ptScs[nowScName]->load();
}

void GlpaBase::loadScene(Glpa::Scene *ptScene)
{
    nowScName = ptScene->getName();
    ptScene->load();
}

void GlpaBase::releaseScene()
{
    ptScs[nowScName]->release();
}

void GlpaBase::releaseScene(Glpa::Scene *ptScene)
{
    ptScene->release();
}

void GlpaBase::releaseAllScene()
{
    for (auto& sc : ptScs)
    {
        sc.second->release();
    }
}

void GlpaBase::setFirstSc(Glpa::Scene *ptScene)
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
