#include "GlpaBase.h"

void GlpaBase::start()
{
    pScs[nowScName]->start();
    started = true;

    window->sendPaintMsg();
}

void GlpaBase::update()
{
    pScs[nowScName]->update();

    window->sendPaintMsg();
}
