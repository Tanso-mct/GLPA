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
    delete instance->fileDataManager;

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

    fileDataManager = new Glpa::FileDataManager();
}

GlpaLib::~GlpaLib()
{

}

LRESULT CALLBACK GlpaLib::WindowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                instance->minimizeMsg(instance->pBcs
                [
                    std::distance
                    (
                        instance->bcsHWnds.begin(), 
                        std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                    )
                ]);
            }
            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_GETMINMAXINFO:
            instance->editSizeMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ], lParam);
            return 0;

        case WM_SETFOCUS:
            instance->focusMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_KILLFOCUS:
            instance->killFocusMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_CREATE:
            instance->createMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_PAINT:
            instance->paintMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_MOVE:
            instance->paintMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_CLOSE:
            instance->closeMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;
                

        case WM_DESTROY:
            instance->destroyMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ]);
            return 0;

        case WM_SYSKEYDOWN:
        case WM_KEYDOWN:
            instance->keyDownMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
                )
            ], msg, wParam, lParam);
            return 0;

        case WM_SYSKEYUP:
        case WM_KEYUP:
            instance->keyUpMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
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
            instance->mouseMsg(instance->pBcs
            [
                std::distance
                (
                    instance->bcsHWnds.begin(), 
                    std::find(instance->bcsHWnds.begin(), instance->bcsHWnds.end(), hWnd)
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

void GlpaLib::focusMsg(GlpaBase *bc)
{
    bc->setFocusing(true);
}

void GlpaLib::killFocusMsg(GlpaBase *bc)
{
    bc->setFocusing(false);
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
    pBc->setManager(instance->fileDataManager);
    pBc->setup();
    instance->pBcs.push_back(pBc);
}

void GlpaLib::deleteBase(GlpaBase *pBc)
{
    pBc->DeleteAllScene();

    instance->bcsHWnds.erase
    (
        instance->bcsHWnds.begin() + std::distance
        (
            instance->pBcs.begin(), 
            std::find(instance->pBcs.begin(), instance->pBcs.end(), pBc)
        )
    );

    instance->pBcs.erase
    (
        instance->pBcs.begin() + std::distance
        (
            instance->pBcs.begin(), 
            std::find(instance->pBcs.begin(), instance->pBcs.end(), pBc)
        )
    );
    delete pBc;
}

void GlpaLib::CreateWindowNotApi(GlpaBase *pBc)
{
    pBc->window->apiClass.lpfnWndProc = *GlpaLib::WindowProc;
    pBc->window->create(instance->hInstance);
    instance->bcsHWnds.push_back(pBc->window->hWnd);
}

void GlpaLib::ShowWindowNotApi(GlpaBase *pBc, int type)
{
    ShowWindow(pBc->window->hWnd, type);
}

void GlpaLib::Load(GlpaBase *pBc)
{
    pBc->LoadScene();
}

void GlpaLib::release(GlpaBase *pBc)
{
    pBc->ReleaseAllScene();
}

void GlpaLib::Run()
{
    while (true) {
        for (int i = 0; i < instance->pBcs.size(); i++) {
            // Process paint messages first if there are any
            if 
            (
                PeekMessage(&instance->msg, NULL, WM_PAINT, WM_PAINT, PM_REMOVE) && 
                instance->pBcs[i]->getVisible() && instance->pBcs[i]->getStarted()
            ){
                TranslateMessage(&instance->msg);
                DispatchMessage(&instance->msg);
                continue; // Skip the rest of the loop iteration to prioritize paint messages
            }
        }
        
        // Process other messages
        if (PeekMessage(&instance->msg, NULL, 0, 0, PM_REMOVE)) {
            if (instance->msg.message == WM_QUIT) {
                break;
            }
            TranslateMessage(&instance->msg);
            DispatchMessage(&instance->msg);
        }

        // Process your custom updates here
        for (int i = 0; i < instance->pBcs.size(); i++) {
            instance->paintMsg(instance->pBcs[i]);
        }
    }
}

void GlpaLib::createMsg(GlpaBase *bc)
{
    
}

void GlpaLib::paintMsg(GlpaBase *bc)
{
    if(bc->getVisible() && bc->getStarted() && bc->IsAnySceneLoaded())
    {
        bc->runUpdate();
    }
    else if(bc->getVisible() && !bc->getStarted() && bc->IsAnySceneLoaded())
    {
        bc->runStart();
    }
}

void GlpaLib::closeMsg(GlpaBase *bc)
{
    GlpaLib::release(bc);
    DestroyWindow(bc->window->hWnd);
}

void GlpaLib::destroyMsg(GlpaBase *bc)
{
    GlpaLib::deleteBase(bc);
    if (instance->pBcs.size() == 0) PostQuitMessage(0);
}
