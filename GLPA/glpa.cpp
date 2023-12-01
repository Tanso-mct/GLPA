#include "glpa.h"

void Glpa::createWindow
(
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
)
{
    Window newWnd
    (
        wndName, wndApiClassName, wndWidth, wndHeight, wndDpi, wndMaxFps, wndFullScreen,
        wndStyle, loadIcon, loadCursor, backgroundColor, smallIcon
    );
    window.emplace(wndName, newWnd);

    windowApi.registerClass(wndName, window);
    windowApi.createWindow(wndName, window);
}

void Glpa::showWindow(LPCWSTR wndName)
{
    ShowWindow(window[wndName].hWnd, windowApi.nCmdShow);
    UpdateWindow(window[wndName].hWnd);
}
