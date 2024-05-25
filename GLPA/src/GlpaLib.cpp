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

LRESULT GlpaLib::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                pBcs[bcHWnds[hWnd]]->minimizeWindow();
            }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_KILLFOCUS:
            pBcs[bcHWnds[hWnd]]->killFocusWindow();
            return 0;

        case WM_SETFOCUS:
            pBcs[bcHWnds[hWnd]]->setFocusWindow();
            return 0;

        case WM_GETMINMAXINFO:
            pBcs[bcHWnds[hWnd]]->editSizeWindow();
            return 0;

        case WM_CREATE:
            pBcs[bcHWnds[hWnd]]->createWindow();
            return 0;

        case WM_PAINT:
            pBcs[bcHWnds[hWnd]]->paintWindow();
            return 0;

        case WM_CLOSE:
            pBcs[bcHWnds[hWnd]]->closeWindow();
            return 0;
                

        case WM_DESTROY:
            pBcs[bcHWnds[hWnd]]->destroyWindow();
            return 0;

        case WM_KEYDOWN:
            pBcs[bcHWnds[hWnd]]->keyDownMsg(msg, wParam, lParam);
            return 0;

        case WM_KEYUP:
            pBcs[bcHWnds[hWnd]]->keyUpMsg(msg, wParam, lParam);
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
            pBcs[bcHWnds[hWnd]]->mouseMsg(msg, wParam, lParam);
            return 0;
            
        default:
                return DefWindowProc(hWnd, msg, wParam, lParam);
    }
    return 0;
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
