#include "GlpaBase.h"

GlpaBase::GlpaBase()
{
    awake();
}

GlpaBase::~GlpaBase()
{
    destroy();
}

void GlpaBase::addScene(Glpa::Scene *ptScene)
{
    ptScene->setup();
    ptScs.emplace(ptScene->getName(), ptScene);
}

void GlpaBase::deleteScene(Glpa::Scene *ptScene)
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
    ptScene->load();
}

void GlpaBase::releaseScene(Glpa::Scene *ptScene)
{
    ptScene->release();
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
