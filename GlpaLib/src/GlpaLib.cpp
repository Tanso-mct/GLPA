#include "GlpaLib.h"

GlpaLib* GlpaLib::instance = nullptr;

void GlpaLib::Start
(
    const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, 
    const LPSTR arg_lpCmdLine, const int arg_nCmdShow
){
    instance = new GlpaLib(arg_hInstance, arg_hPrevInstance, arg_lpCmdLine, arg_nCmdShow);
}

int GlpaLib::Close()
{
    MSG rtMsg = instance->msg;
    delete instance;

    return static_cast<int>(rtMsg.wParam);
}

GlpaLib::GlpaLib(
    const HINSTANCE arg_hInstance, const HINSTANCE arg_hPrevInstance, const LPSTR arg_lpCmdLine, const int arg_nCmdShow)
{
    hInstance = arg_hInstance;
    hPrevInstance = arg_hInstance;
    lpCmdLine = arg_lpCmdLine;
    nCmdShow = arg_nCmdShow;
}

GlpaLib::~GlpaLib()
{

}

LRESULT CALLBACK GlpaLib::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                GlpaLib::instance->minimizeMsg(GlpaLib::instance->pBcs
                [
                    std::distance
                    (
                        GlpaLib::instance->bcsHWnds.begin(), 
                        std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                    )
                ]);
            }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_GETMINMAXINFO:
            GlpaLib::instance->editSizeMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ], lParam);
            return 0;

        case WM_CREATE:
            GlpaLib::instance->createMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_PAINT:
            GlpaLib::instance->paintMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_CLOSE:
            GlpaLib::instance->closeMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;
                

        case WM_DESTROY:
            GlpaLib::instance->destroyMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            GlpaLib::instance->keyDownMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ], msg, wParam, lParam);
            return 0;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            GlpaLib::instance->keyUpMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ], msg, wParam, lParam);
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
            GlpaLib::instance->mouseMsg(GlpaLib::instance->pBcs
            [
                std::distance
                (
                    GlpaLib::instance->bcsHWnds.begin(), 
                    std::find(GlpaLib::instance->bcsHWnds.begin(), GlpaLib::instance->bcsHWnds.end(), hWnd)
                )
            ], msg, wParam, lParam);
            return 0;
            
        default:
            return DefWindowProc(hWnd, msg, wParam, lParam);
    }
    return 0;
}

void GlpaLib::minimizeMsg(GlpaBase *bc)
{
    GlpaLib::ShowWindowNotApi(bc, SW_MINIMIZE);
}

void GlpaLib::editSizeMsg(GlpaBase *bc, LPARAM lParam)
{
    MINMAXINFO* pMinMaxInfo = (MINMAXINFO*)lParam;
    pMinMaxInfo->ptMinTrackSize.x = bc->window->GetWidth();
    pMinMaxInfo->ptMinTrackSize.y = bc->window->GetHeight();
    pMinMaxInfo->ptMaxTrackSize.x = bc->window->GetWidth();
    pMinMaxInfo->ptMaxTrackSize.y = bc->window->GetHeight();
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
    bc->getNowScenePt()->getMouse(msg, wParam, lParam, bc->window->GetDpi());
}

void GlpaLib::AddBase(GlpaBase *pBc)
{
    pBc->setup();
    GlpaLib::instance->pBcs.push_back(pBc);
}

void GlpaLib::DeleteBase(GlpaBase *pBc)
{
    pBc->DeleteAllScene();

    GlpaLib::instance->bcsHWnds.erase
    (
        GlpaLib::instance->bcsHWnds.begin() + std::distance
        (
            GlpaLib::instance->pBcs.begin(), 
            std::find(GlpaLib::instance->pBcs.begin(), GlpaLib::instance->pBcs.end(), pBc)
        )
    );

    GlpaLib::instance->pBcs.erase
    (
        GlpaLib::instance->pBcs.begin() + std::distance
        (
            GlpaLib::instance->pBcs.begin(), 
            std::find(GlpaLib::instance->pBcs.begin(), GlpaLib::instance->pBcs.end(), pBc)
        )
    );
    delete pBc;
}

void GlpaLib::CreateWindowNotApi(GlpaBase *pBc)
{
    pBc->window->apiClass.lpfnWndProc = *GlpaLib::WindowProc;
    pBc->window->create(GlpaLib::instance->hInstance);
    GlpaLib::instance->bcsHWnds.push_back(pBc->window->hWnd);
}

void GlpaLib::ShowWindowNotApi(GlpaBase *pBc, int type)
{
    ShowWindow(pBc->window->hWnd, type);
}

void GlpaLib::Load(GlpaBase *pBc)
{
    pBc->LoadScene();
}

void GlpaLib::Release(GlpaBase *pBc)
{
    pBc->ReleaseAllScene();
}

void GlpaLib::Run()
{
    while (true) {
        if (PeekMessage(&GlpaLib::instance->msg, NULL, 0, 0, PM_REMOVE)) {
            if (GlpaLib::instance->msg.message == WM_QUIT) {
                break;
            }
            TranslateMessage(&GlpaLib::instance->msg);
            DispatchMessage(&GlpaLib::instance->msg);
        } 

        for (int i = 0; i < GlpaLib::instance->pBcs.size(); i++) {
            if(GlpaLib::instance->pBcs[i]->getVisible() && GlpaLib::instance->pBcs[i]->getStarted())
            {
                GlpaLib::instance->pBcs[i]->runUpdate();
            }
            else if(GlpaLib::instance->pBcs[i]->getVisible() && !GlpaLib::instance->pBcs[i]->getStarted())
            {
                GlpaLib::instance->pBcs[i]->runStart();
            }
        }
    }
}

void GlpaLib::createMsg(GlpaBase *bc)
{
    
}

void GlpaLib::paintMsg(GlpaBase *bc)
{
    bc->window->paint();
}

void GlpaLib::closeMsg(GlpaBase *bc)
{
    GlpaLib::Release(bc);
    DestroyWindow(bc->window->hWnd);
}

void GlpaLib::destroyMsg(GlpaBase *bc)
{
    GlpaLib::DeleteBase(bc);
    if (GlpaLib::instance->pBcs.size() == 0) PostQuitMessage(0);
}
