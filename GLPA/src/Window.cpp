#include "Window.h"

void Glpa::Window::createPixels()
{
    pixels = (LPDWORD)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, width * height * 4 * dpi);
}

void Glpa::Window::create(HINSTANCE hInstance)
{
    apiClass.cbSize = sizeof(apiClass);
    apiClass.style = style;
    apiClass.cbClsExtra = NULL;
    apiClass.cbWndExtra = NULL;
    apiClass.hInstance = hInstance;

    apiClass.hIcon = (HICON)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadIcon),
        IMAGE_ICON,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );

    apiClass.hCursor = (HCURSOR)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadCursor),
        IMAGE_CURSOR,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );     
                                                
    apiClass.hbrBackground = (HBRUSH)GetStockObject(bgColor);
    apiClass.lpszMenuName = NULL;
    apiClass.lpszClassName = apiClassName;
    apiClass.hIconSm =
    LoadIcon(apiClass.hInstance, MAKEINTRESOURCE(smallIcon));

    if (!RegisterClassEx(&apiClass)){
        OutputDebugStringA("ERROR Window.cpp - Class registration failed.\n");
        throw std::runtime_error("Class registration failed.");
    }


    hWnd = CreateWindow(
        apiClassName,
        name,
        viewStyle,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (!hWnd){
        OutputDebugStringA("ERROR Window.cpp - Failed to create window.\n");
        throw std::runtime_error("Failed to create window.");
    }
}

void Glpa::Window::createDc()
{
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
        (LPVOID*)&pixels, 
        NULL, 
        0
    );
    SelectObject(hBufDC, hBufBmp);

    ReleaseDC(hWnd, hWndDC);
}

void Glpa::Window::paint()
{
    hWndDC = BeginPaint(hWnd, &hPs);

    StretchDIBits(
        hWndDC,
        0,
        0,
        width,
        height,
        0,
        0,
        width * dpi,
        height * dpi, 
        pixels,
        &hBufBmpInfo,
        DIB_RGB_COLORS,
        SRCCOPY
    );
    
    EndPaint(hWnd, &hPs);
}

void Glpa::Window::sendPaintMsg()
{
    InvalidateRect(hWnd, NULL, FALSE);
}
