#include "GlpaBase.h"

GlpaBase::GlpaBase()
{
    awake();
}

GlpaBase::~GlpaBase()
{
    destroy();
}

void GlpaBase::runStart()
{
    start();
    pScs[nowScName]->start();
    started = true;

    window->sendPaintMsg();
}

void GlpaBase::runUpdate()
{
    update();
    pScs[nowScName]->update();

    window->sendPaintMsg();
}
