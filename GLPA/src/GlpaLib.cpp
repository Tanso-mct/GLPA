#include "GlpaLib.h"

GlpaLib::GlpaLib
(
    const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, const LPSTR arg_lpCmdLine, const int arg_nCmdShow
)
{
    hInstance = arg_hInstance;
    hPrevInstance = arg_hInstance;
    lpCmdLine = arg_lpCmdLine;
    nCmdShow = arg_nCmdShow;
}

void GlpaLib::addBase(GlpaBase *pBc)
{
    pBcs.emplace(pBc->getName(), pBc);
}

void GlpaLib::deleteBase(GlpaBase *pBc)
{
    delete pBcs[pBc->getName()];
    pBcs.erase(pBc->getName());
}

void GlpaLib::run()
{
    while (true) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } 

        for (auto& pBc : pBcs) {
            if(pBc.second->getVisible() && !pBc.second->getStarted())
            {
                pBc.second->start();
            }
            else if(pBc.second->getVisible() && pBc.second->getStarted())
            {
                pBc.second->update();
            }
        }
    }
}
