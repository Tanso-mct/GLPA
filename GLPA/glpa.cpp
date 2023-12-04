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
    double wndWidth,
    double wndHeight,
    double wndDpi,
    double wndMaxFps,
    UINT wndStyle,
    LPWSTR loadIcon, 
    LPWSTR loadCursor,
    int backgroundColor,
    LPWSTR smallIcon
){

    Window newWnd
    (
        wndName, wndApiClassName, wndWidth, wndHeight, wndDpi, wndMaxFps,
        wndStyle, loadIcon, loadCursor, backgroundColor, smallIcon
    );

    window.emplace(wndName, newWnd);
    window[wndName].create(hInstance, ptWindowProc);
}

void Glpa::showWindow(LPCWSTR wndName)
{
    window[wndName].show();
}

void Glpa::updateWindowInfo(LPCWSTR wndName)
{
    window[wndName].changeSize();
}

Glpa glpa;

LRESULT CALLBACK windowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){
        switch (msg)
        {
        case WM_KILLFOCUS:
                return 0;

        case WM_CREATE :
                for (auto& x: glpa.window){
                    if (x.second.createMsg()){
                        break;
                    }
                }
                return 0;

        case WM_PAINT :
                return 0;

        case WM_CLOSE :
                for (auto& x: glpa.window) {
                    if(hWnd == x.second.createdHWND){
                        OutputDebugStringW(_T("Created\n"));
                    }
                    else{
                        OutputDebugStringW(_T("Failed\n"));
                    }
                }

                DestroyWindow(hWnd);
                return 0;
                

        case WM_DESTROY :
                PostQuitMessage(0);
                return 0;


        default :
                return DefWindowProc(hWnd, msg, wParam, lParam);
        }
        return 0;
}