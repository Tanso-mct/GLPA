#include "window.h"

void Window::create(HINSTANCE arghInstance, WINDOW_PROC_TYPE* ptWindowProc){
    wndClass.cbSize = sizeof(wndClass);
    wndClass.style = style;
    wndClass.lpfnWndProc = *ptWindowProc;
    wndClass.cbClsExtra = NULL;
    wndClass.cbWndExtra = NULL;
    wndClass.hInstance = arghInstance;

    wndClass.hIcon = (HICON)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadIcon),
        IMAGE_ICON,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );

    wndClass.hCursor = (HCURSOR)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadCursor),
        IMAGE_CURSOR,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );     
                                                
    wndClass.hbrBackground = (HBRUSH)GetStockObject(backgroundColor);
    wndClass.lpszMenuName = NULL;
    wndClass.lpszClassName = nameApiClass;
    wndClass.hIconSm =
    LoadIcon(wndClass.hInstance, MAKEINTRESOURCE(smallIcon));

    if (!RegisterClassEx(&wndClass)){
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            name,
            MB_ICONEXCLAMATION
        );
    }

    hWnd = CreateWindow(
        nameApiClass,
        name,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        arghInstance,
        NULL
    );

    if (!hWnd){
        MessageBox(
            NULL,
            _T("window make fail"),
            name,
            MB_ICONEXCLAMATION
        );
    }
}

void Window::show(){
    ShowWindow(hWnd, SW_SHOWDEFAULT);
}

void Window::hide(){
    ShowWindow(hWnd, SW_HIDE);
}

void Window::changeSize(){
    SetWindowPos(hWnd, NULL, 0, 0, 500, 500, SWP_NOMOVE | SWP_NOZORDER);
}

bool Window::createMsg(){
    if (createdHWND == nullptr)
    {
        createdHWND = hWnd;

        // Create buffer
        
        return true;
    }

    return false;
}