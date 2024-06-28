#include "Window.h"

Glpa::Window::~Window()
{
    releaseD2D();
    delete pixels;
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
        OutputDebugStringA("GlpaLib ERROR Window.cpp - Class registration failed.\n");
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
        OutputDebugStringA("GlpaLib ERROR Window.cpp - Failed to create window.\n");
        throw std::runtime_error("Failed to create window.");
    }

    pixels = new DWORD[width * height * dpi];

    initD2D();
}

void Glpa::Window::initD2D()
{
    HRESULT hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pFactory);
    if (SUCCEEDED(hr))
    {
        RECT rc;
        GetClientRect(hWnd, &rc);

        hr = pFactory->CreateHwndRenderTarget(
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(hWnd, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
            &pRenderTarget
        );

        if (FAILED(hr))
        {
            OutputDebugStringA("GlpaLib ERROR Window.cpp - Failed to create direct2d render target.\n");
            throw std::runtime_error("Failed to create direct2d render target.");
            return;
        }
    }
    else
    {
        OutputDebugStringA("GlpaLib ERROR Window.cpp - Failed to create direct2d factory.\n");
        throw std::runtime_error("Failed to create direct2d factory.");
        return;
    }
}

void Glpa::Window::releaseD2D()
{
    if (pBitmap != nullptr) pBitmap->Release();
    if (pRenderTarget != nullptr) pRenderTarget->Release();
    if (pFactory != nullptr) pFactory->Release();
}

void Glpa::Window::paint()
{
    OutputDebugStringA("PAINT\n");

    BeginPaint(hWnd, &hPs);
    pRenderTarget->BeginDraw();
    
    D2D1_SIZE_F size = pBitmap->GetSize();
    pRenderTarget->DrawBitmap(pBitmap, D2D1::RectF(0.0f, 0.0f, size.width, size.height));
    
    pRenderTarget->EndDraw();
    EndPaint(hWnd, &hPs);

}

void Glpa::Window::SetWidth(int value)
{
    width = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::SetHeight(int value)
{
    height = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::SetDpi(int value)
{
    dpi = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::setViewStyle(UINT value)
{
    if (hWnd == nullptr)
    {
        viewStyle = value;
    }
    else
    {
        viewStyle = value;
        SetWindowLongPtr(hWnd, GWL_STYLE, viewStyle);
    }
}

void Glpa::Window::addViewStyle(UINT value)
{
    if (hWnd == nullptr)
    {
        viewStyle |= value;
    }
    else
    {
        viewStyle |= value;
        SetWindowLongPtr(hWnd, GWL_STYLE, viewStyle);
    }
}

void Glpa::Window::deleteViewStyle(UINT value)
{
    if (hWnd == nullptr)
    {
        viewStyle &= ~value;
    }
    else
    {
        viewStyle &= ~value;
        SetWindowLongPtr(hWnd, GWL_STYLE, viewStyle);
    }
}

void Glpa::Window::sendPaintMsg()
{
    InvalidateRect(hWnd, NULL, FALSE);
}
