#include "window_api.h"

void WindowApi::registerClass(LPCWSTR wndName, std::unordered_map<LPCWSTR, Window> window)
{
    WNDCLASSEX wndClass;
    wndClass.cbSize = sizeof(wndClass);
    wndClass.style = window[wndName].style;
    wndClass.lpfnWndProc = window[wndName].procedure;
    wndClass.cbClsExtra = NULL;
    wndClass.cbWndExtra = NULL;
    wndClass.hInstance = hInstance;
    wndClass.hIcon = (HICON)LoadImage
    (
        NULL, 
        MAKEINTRESOURCE(window[wndName].loadIcon),
        IMAGE_ICON,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );
    wndClass.hCursor = (HCURSOR)LoadImage
    (
        NULL, 
        MAKEINTRESOURCE(window[wndName].loadCursor),
        IMAGE_CURSOR,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );                                                 
    wndClass.hbrBackground = (HBRUSH)GetStockObject(window[wndName].backgroundColor);
    wndClass.lpszMenuName = NULL;
    wndClass.lpszClassName = window[wndName].nameApiClass;
    wndClass.hIconSm =
    LoadIcon(wndClass.hInstance, MAKEINTRESOURCE(window[wndName].smallIcon));

    if (!RegisterClassEx(&wndClass))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            wndName,
            MB_ICONEXCLAMATION
        );
    }
}

void WindowApi::createWindow(LPCWSTR wndName, std::unordered_map<LPCWSTR, Window> window)
{
    window[wndName].hWnd = CreateWindow
    (
        window[wndName].nameApiClass,
        window[wndName].name,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        window[wndName].width, window[wndName].height,
        HWND_DESKTOP,
        NULL,
        hInstance,
        NULL
    );

    if (!window[wndName].hWnd)
    {
        MessageBox(
            NULL,
            _T("window make fail"),
            wndName,
            MB_ICONEXCLAMATION
        );
    }

}