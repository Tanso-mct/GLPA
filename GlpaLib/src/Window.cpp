#include "Window.h"
#include "GlpaLog.h"

Glpa::Window::Window()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::Window::~Window()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor Window[" + strConverter.to_bytes(name) + "]");
    releaseD2D();
    delete pixels;
}

void Glpa::Window::create(HINSTANCE hInstance)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Window[" + strConverter.to_bytes(name) + "]");

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
        Glpa::runTimeError(__FILE__, __LINE__, "Class registration failed.");
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
        Glpa::runTimeError(__FILE__, __LINE__, "Failed to create window.");
    }

    pixels = new DWORD[width * height * dpi];

    initD2D();
}

void Glpa::Window::initD2D()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Window[" + strConverter.to_bytes(name) + "]");
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
            Glpa::runTimeError(__FILE__, __LINE__, "Failed to create direct2d render target.");
            return;
        }
    }
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, "Failed to create direct2d factory");
        return;
    }
}

void Glpa::Window::releaseD2D()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Window[" + strConverter.to_bytes(name) + "]");
    if (pBitmap != nullptr) pBitmap->Release();
    if (pRenderTarget != nullptr) pRenderTarget->Release();
    if (pFactory != nullptr) pFactory->Release();
}

void Glpa::Window::SetWidth(int value)
{
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, 
        Glpa::OUTPUT_TAG_GLPA_WINDOW, "Window[" + strConverter.to_bytes(name) + "] width : " + std::to_string(value)
    );

    width = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::SetHeight(int value)
{
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_WINDOW, 
        "Window[" + strConverter.to_bytes(name) + "] height : " + std::to_string(value)
    );

    height = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::SetDpi(int value)
{
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_WINDOW, 
        "Window[" + strConverter.to_bytes(name) + "] dpi : " + std::to_string(value)
    );

    dpi = value;
    delete pixels;
    pixels = new DWORD[width * height * dpi];
}

void Glpa::Window::setViewStyle(UINT value)
{
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_WINDOW, 
        "Window[" + strConverter.to_bytes(name) + "] viewStyle : " + std::to_string(value)
    );

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
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_WINDOW, 
        "Window[" + strConverter.to_bytes(name) + "] viewStyle : " + std::to_string(value)
    );

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
    Glpa::OutputLog
    (
        __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_WINDOW, 
        "Window[" + strConverter.to_bytes(name) + "] viewStyle : " + std::to_string(value)
    );

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
