#include "window.h"

void Window::getFps(){

}

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

void Window::updateSize(){
    SetWindowPos(hWnd, NULL, 0, 0, 500, 500, SWP_NOMOVE | SWP_NOZORDER);
}

void Window::updateStatus(int argStatus){
    if(argStatus == WINDOW_STATUS_DEF){
        ShowWindow(hWnd, SW_SHOWDEFAULT);
    }
    else if (argStatus == WINDOW_STATUS_HIDE){
        ShowWindow(hWnd, SW_HIDE);
    }
}

bool Window::isVisiable(){
    if (visiable){
        return true;
    }

    return false;
}

void Window::graphicLoop(){
    getFps();

    for(UINT y = 0; y <= height; y++)
    {
        for(UINT x = 0; x <= 200; x++)
        {
            if (x < 200 && y < height)
            {
                lpPixel[x+y*width * dpi] = 0x00FF0000;
            }  
        }
    }

    InvalidateRect(hWnd, NULL, FALSE);
}


bool Window::killFoucusMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        focus = false;
    
        // Obtains coordinate information for the target window.
        RECT rect;
        if (GetWindowRect(hWnd, &rect)) {
            // Determine if the target window is hidden by other windows.
            HWND overlappingWindow = GetWindow(hWnd, GW_HWNDPREV);
            while (overlappingWindow != nullptr) {
                RECT overlapRect;
                if (GetWindowRect(overlappingWindow, &overlapRect)) {
                    // Check for overlapping windows.
                    if (IntersectRect(nullptr, &rect, &overlapRect)) {
                        visiable = false;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool Window::setFoucusMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        focus = true;
        visiable = true;
        OutputDebugStringW(_T("GLPA : FOCUS\n"));
        return true;
    }
    
    return false;
}


bool Window::createMsg(HWND argHWnd){
    if (!created)
    {
        OutputDebugStringW(_T("GLPA : CREATED\n"));

        hWndDC = GetDC(hWnd);

        //bmp buffer dc
        hBufBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        hBufBmpInfo.bmiHeader.biWidth = +width * dpi;
        hBufBmpInfo.bmiHeader.biHeight = -height * dpi;      
        hBufBmpInfo.bmiHeader.biPlanes = 1;
        hBufBmpInfo.bmiHeader.biBitCount = 32;
        hBufBmpInfo.bmiHeader.biCompression = BI_RGB;
        
        hBufDC = CreateCompatibleDC(hWndDC);
        hBufBmp = CreateDIBSection
        (
            NULL, 
            &hBufBmpInfo, 
            DIB_RGB_COLORS, 
            (LPVOID*)&lpPixel, 
            NULL, 
            0
        );
        SelectObject(hBufDC, hBufBmp);

        ReleaseDC(hWnd, hWndDC);

        created = true;
        return true;
    }

    OutputDebugStringW(_T("GLPA : NOT CREATED\n"));
    return false;
}

bool Window::closeMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        DestroyWindow(hWnd);

        OutputDebugStringW(_T("GLPA : CLOSED\n"));
        return true;
    }
    
    OutputDebugStringW(_T("GLPA : NOT CLOSED\n"));
    return false;
}

bool Window::destroyMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        PostQuitMessage(0);

        OutputDebugStringW(_T("GLPA : DESTROYED\n"));
        return true;
    }
    
    OutputDebugStringW(_T("GLPA : NOT DESTROYED\n"));
    return false;
}

bool Window::paintMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        hWndDC = BeginPaint(hWnd, &hPs);

        StretchDIBits(
            hWndDC,
            0,
            0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN), 
            0,
            0,
            width * dpi,
            height * dpi, 
            lpPixel,
            &hBufBmpInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );
        EndPaint(hWnd, &hPs);
        
        return true;
    }
    
    return false;
}

bool Window::userMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        if (focus)
        {

        }
        
        return true;
    }
    
    return false;
}