#include "window_api.h"

void WindowApi::createWindow(LPCWSTR wndName, std::unordered_map<LPCWSTR, Window>* window)
{
    (*window)[wndName].wndClass.cbSize = sizeof((*window)[wndName].wndClass);
    (*window)[wndName].wndClass.style = (*window)[wndName].style;
    (*window)[wndName].wndClass.lpfnWndProc = *ptWindowProc;
    (*window)[wndName].wndClass.cbClsExtra = NULL;
    (*window)[wndName].wndClass.cbWndExtra = NULL;
    (*window)[wndName].wndClass.hInstance = hInstance;

    (*window)[wndName].wndClass.hIcon = (HICON)LoadImage(
        NULL, 
        MAKEINTRESOURCE((*window)[wndName].loadIcon),
        IMAGE_ICON,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );

    (*window)[wndName].wndClass.hCursor = (HCURSOR)LoadImage(
        NULL, 
        MAKEINTRESOURCE((*window)[wndName].loadCursor),
        IMAGE_CURSOR,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );     
                                                
    (*window)[wndName].wndClass.hbrBackground = (HBRUSH)GetStockObject((*window)[wndName].backgroundColor);
    (*window)[wndName].wndClass.lpszMenuName = NULL;
    (*window)[wndName].wndClass.lpszClassName = (*window)[wndName].nameApiClass;
    (*window)[wndName].wndClass.hIconSm =
    LoadIcon((*window)[wndName].wndClass.hInstance, MAKEINTRESOURCE((*window)[wndName].smallIcon));

    if (!RegisterClassEx(&(*window)[wndName].wndClass)){
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            wndName,
            MB_ICONEXCLAMATION
        );
    }

    (*window)[wndName].hWnd = CreateWindow(
        (*window)[wndName].nameApiClass,
        (*window)[wndName].name,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        (*window)[wndName].width, (*window)[wndName].height,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (!(*window)[wndName].hWnd){
        MessageBox(
            NULL,
            _T("window make fail"),
            wndName,
            MB_ICONEXCLAMATION
        );
    }
}

void WindowApi::showWindow(LPCWSTR wndName, std::unordered_map<LPCWSTR, Window> *window)
{
    ShowWindow((*window)[wndName].hWnd, SW_SHOWDEFAULT);
}

