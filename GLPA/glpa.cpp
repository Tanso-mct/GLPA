#include "glpa.h"

void Glpa::initialize(HINSTANCE arghInstance, HINSTANCE arghPrevInstance, LPSTR arglpCmdLine, int argnCmdShow)
{
    hInstance = arghInstance;
    hPrevInstance = arghPrevInstance;
    lpCmdLine = arglpCmdLine;
    nCmdShow = argnCmdShow;
    ptWindowProc = windowProc;
}

void Glpa::createWindow(
    LPCWSTR wndName,
    LPCWSTR wndApiClassName,
    int wndWidth,
    int wndHeight,
    int wndDpi,
    double wndMaxFps,
    UINT wndStyle,
    LPWSTR loadIcon, 
    LPWSTR loadCursor,
    int backgroundColor,
    LPWSTR smallIcon,
    bool minimizeAuto
){

    Window newWnd
    (
        wndName, wndApiClassName, wndWidth, wndHeight, wndDpi, wndMaxFps,
        wndStyle, loadIcon, loadCursor, backgroundColor, smallIcon, minimizeAuto
    );

    window.emplace(wndName, newWnd);
    window[wndName].create(hInstance, ptWindowProc);
}

void Glpa::updateWindow(LPCWSTR wndName, int param){
    switch (param)
    {
    case WINDOW_STATUS_DEF :
        window[wndName].updateStatus(WINDOW_STATUS_DEF);
        break;

    case WINDOW_STATUS_HIDE :
        window[wndName].updateStatus(WINDOW_STATUS_HIDE);
        break;

    case WINDOW_STATUS_MINIMIZE :
        window[wndName].updateStatus(WINDOW_STATUS_MINIMIZE);
        break;

    default:
        OutputDebugStringW(_T(ERROR_ARUGUMENT_INCOLLECT));
        OutputDebugStringW(_T("Glpa::updateWindow(LPCWSTR wndName, int param) -> int param"));
        break;
    }

}

void Glpa::runGraphicLoop(){
    while (true) {
        // Returns 1 (true) if a message is retrieved and 0 (false) if not.
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                OutputDebugStringW(_T("GLPA : EXIT\n"));

                // Exit from the loop when the exit message comes.
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } 

        for (auto& x: window) {
            if(x.second.isVisiable()){
                x.second.graphicLoop();
                break;
            }
        }
            // else if (WndPLAY.state.focus)
            // {
            //     WndPLAY.fpsSystem.fpsLimiter();

            //     PatBlt(
            //         WndPLAY.buffer.hBufDC, 
            //         0, 
            //         0, 
            //         WINDOW_WIDTH * DISPLAY_RESOLUTION, 
            //         WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
            //         WHITENESS
            //     );
                // scrPLAYDwgContModif(WndPLAY.buffer.hBufDC);

            //     InvalidateRect(WndPLAY.hWnd, NULL, FALSE);
            // }
            // else if (WndLAU.state.focus)
            // {
            //     WndLAU.fpsSystem.fpsLimiter();

            //     PatBlt(
            //         WndLAU.buffer.hBufDC, 
            //         0, 
            //         0, 
            //         WINDOW_WIDTH * DISPLAY_RESOLUTION, 
            //         WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
            //         WHITENESS
            //     );
            //     scrLAUDwgContModif(WndLAU.buffer.hBufDC);

            //     InvalidateRect(WndLAU.hWnd, NULL, FALSE);
            // }
            
    }
}

Glpa glpa;

LRESULT CALLBACK windowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){
        switch (msg){
        case WM_SYSCOMMAND:
            if (wParam == SC_MINIMIZE) {
                for (auto& x: glpa.window) {
                    if(x.second.minimizeMsg(hWnd)){
                        return 0;
                    }
                }
            }

            return DefWindowProc(hWnd, msg, wParam, lParam);

        case WM_KILLFOCUS:
                for (auto& x: glpa.window) {
                    if(x.second.killFoucusMsg(hWnd)){
                        break;
                    }
                }
                return 0;
 
        case WM_SETFOCUS:
                for (auto& x: glpa.window) {
                    if(x.second.setFoucusMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_CREATE :
                for (auto& x: glpa.window){
                    if (x.second.createMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_PAINT :
                for (auto& x: glpa.window) {
                    if(x.second.paintMsg(hWnd)){
                        break;
                    }
                }
        
                return 0;

        case WM_CLOSE :
                for (auto& x: glpa.window) {
                    if(x.second.closeMsg(hWnd)){
                        break;
                    }
                }
                return 0;
                

        case WM_DESTROY :
                for (auto& x: glpa.window) {
                    if(x.second.destroyMsg(hWnd)){
                        break;
                    }
                }
                return 0;

        case WM_KEYDOWN :
        case WM_KEYUP :
        case WM_LBUTTONDOWN :
        case WM_LBUTTONUP :
        case WM_LBUTTONDBLCLK :
        case WM_RBUTTONDOWN :
        case WM_RBUTTONUP :
        case WM_RBUTTONDBLCLK :
        case WM_MBUTTONDOWN :
        case WM_MBUTTONDBLCLK :
        case WM_MBUTTONUP :
        case WM_MOUSEWHEEL :
                for (auto& x: glpa.window) {
                    if(x.second.userMsg(hWnd)){
                        break;
                    }
                }
                return 0;



        default :
                return DefWindowProc(hWnd, msg, wParam, lParam);
        }
        return 0;
}