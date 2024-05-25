#include "GlpaBase.h"

void GlpaBase::keyDownMsg(UINT msg, WPARAM wParam, LPARAM lParam)
{
    pScs[nowScName]->getKeyDown(msg, wParam, lParam);
}

void GlpaBase::keyUpMsg(UINT msg, WPARAM wParam, LPARAM lParam)
{
    pScs[nowScName]->getKeyUp(msg, wParam, lParam);
}

void GlpaBase::mouseMsg(UINT msg, WPARAM wParam, LPARAM lParam)
{
    pScs[nowScName]->getMouse(msg, wParam, lParam);
}

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
