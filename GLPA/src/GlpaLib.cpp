#include "GlpaLib.h"

void GlpaLib::start
(
    const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
    const LPSTR arg_lpCmdLine, const int arg_nCmdShow
){
    instance = new GlpaLib(arg_hInstance, arg_hPrevInstance, arg_lpCmdLine, arg_nCmdShow);
}

void GlpaLib::close()
{
    delete instance;
}

GlpaLib::GlpaLib(
    const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, const LPSTR arg_lpCmdLine, const int arg_nCmdShow)
{
    hInstance = arg_hInstance;
    hPrevInstance = arg_hInstance;
    lpCmdLine = arg_lpCmdLine;
    nCmdShow = arg_nCmdShow;
}

LRESULT CALLBACK GlpaLib::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                GlpaLib::instance->minimizeMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_KILLFOCUS:
            GlpaLib::instance->killFocusMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_SETFOCUS:
            GlpaLib::instance->setFocusMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_GETMINMAXINFO:
            GlpaLib::instance->editSizeMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_CREATE:
            GlpaLib::instance->createMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_PAINT:
            GlpaLib::instance->paintMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_CLOSE:
            GlpaLib::instance->closeMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;
                

        case WM_DESTROY:
            GlpaLib::instance->destroyMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]]);
            return 0;

        case WM_KEYDOWN:
            GlpaLib::instance->keyDownMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]], msg, wParam, lParam);
            return 0;

        case WM_KEYUP:
            GlpaLib::instance->keyUpMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]], msg, wParam, lParam);
            return 0;

        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_LBUTTONDBLCLK:
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_RBUTTONDBLCLK:
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:
        case WM_MOUSEMOVE:
            GlpaLib::instance->mouseMsg(GlpaLib::instance->pBcs[GlpaLib::instance->bcHWnds[hWnd]], msg, wParam, lParam);
            return 0;
            
        default:
                return DefWindowProc(hWnd, msg, wParam, lParam);
    }
    return 0;
}

void GlpaLib::keyDownMsg(GlpaBase *bc, UINT msg, WPARAM wParam, LPARAM lParam)
{
    bc->getNowScenePt()->getKeyDown(msg, wParam, lParam);
}

void GlpaLib::keyUpMsg(GlpaBase *bc, UINT msg, WPARAM wParam, LPARAM lParam)
{
    bc->getNowScenePt()->getKeyUp(msg, wParam, lParam);
}

void GlpaLib::mouseMsg(GlpaBase *bc, UINT msg, WPARAM wParam, LPARAM lParam)
{
    bc->getNowScenePt()->getMouse(msg, wParam, lParam);
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

void GlpaLib::createWindow(GlpaBase *pBc)
{
    Glpa::Window* ptWindow = pBc->window;

    pBc->window->apiClass.lpfnWndProc = *GlpaLib::WindowProc;
    ptWindow->create(hInstance);
}

void GlpaLib::showWindow(GlpaBase *pBc, int type)
{
    ShowWindow(pBc->window->hWnd, type);
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
            if(pBc.second->getVisible() && pBc.second->getStarted())
            {
                pBc.second->runUpdate();
            }
            else if(pBc.second->getVisible() && !pBc.second->getStarted())
            {
                pBc.second->runStart();
            }
        }
    }
}

void GlpaLib::createMsg(GlpaBase *bc)
{
    bc->window->createDc();
}
