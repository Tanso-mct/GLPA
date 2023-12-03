#include "glpa.h"

void Glpa::initialize(HINSTANCE arghInstance, HINSTANCE arghPrevInstance, LPSTR arglpCmdLine, int argnCmdShow)
{
    windowsApi.hInstance = arghInstance;
    windowsApi.hPrevInstance = arghPrevInstance;
    windowsApi.lpCmdLine = arglpCmdLine;
    windowsApi.nCmdShow = argnCmdShow;
}

void Glpa::createWindow(
    LPCWSTR wndName,
    LPCWSTR wndApiClassName,
    double wndWidth,
    double wndHeight,
    double wndDpi,
    double wndMaxFps,
    bool wndFullScreen,
    UINT wndStyle,
    LPWSTR loadIcon, 
    LPWSTR loadCursor,
    int backgroundColor,
    LPWSTR smallIcon
){
    windowsApi.ptWindowProc = windowProc;

    Window newWnd
    (
        wndName, wndApiClassName, wndWidth, wndHeight, wndDpi, wndMaxFps, wndFullScreen,
        wndStyle, loadIcon, loadCursor, backgroundColor, smallIcon
    );

    window.emplace(wndName, newWnd);
    windowsApi.createWindow(wndName, &window);
}

void Glpa::showWindow(LPCWSTR wndName)
{
    windowsApi.showWindow(wndName, &window);
}

void Glpa::updateWindowInfo(LPCWSTR wndName)
{
    SetWindowPos(window[wndName].hWnd, NULL, 0, 0, 500, 500, SWP_NOMOVE | SWP_NOZORDER);
}

Glpa glpa;

LRESULT CALLBACK windowProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam){
        switch (msg)
        {
        case WM_KILLFOCUS:
                return 0;

        case WM_CREATE :
                for (auto& x: glpa.window) {
                    if (x.second.createdHWND == nullptr)
                    {
                        x.second.createdHWND = hWnd;
                        x.second.create();
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