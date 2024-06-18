#include <Windows.h>
#include <d2d1.h>

#include <atlcomcli.h>

#pragma comment(lib, "d2d1")

#define WIDTH 640
#define HEIGHT 480

using D2D1::ColorF;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

HRESULT CreateResource();
void OnPain();

HWND m_hwnd = {};

CComPtr<ID2D1Factory> factory;
CComPtr<ID2D1HwndRenderTarget> rt;
CComPtr<ID2D1SolidColorBrush> brush;

bool keyRight = false;
bool keyLeft = false;
bool keyUp = false;
bool keyDown = false;

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR pCmd, int nCmd)
{
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"Window Class";
    RegisterClass(&wc);

    RECT rc = {0, 0, WIDTH, HEIGHT};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    m_hwnd = CreateWindowEx
    (
        0, wc.lpszClassName, L"MOVE CIRCLE",
        WS_OVERLAPPEDWINDOW ^ WS_SIZEBOX ^ WS_MAXIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
        NULL, NULL, hInst, NULL
    );
    if (m_hwnd == NULL) return 0;

    ShowWindow(m_hwnd, nCmd);

    MSG msg = {};
    while(GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CREATE:
        if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &factory)))
        {
            return -1;
        }

        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_KEYDOWN:
        if (wParam == VK_RIGHT) keyRight = true;
        if (wParam == VK_LEFT) keyLeft = true;
        if (wParam == VK_UP) keyUp = true;
        if (wParam == VK_DOWN) keyDown = true;
        return 0;

    case WM_KEYUP:
        if (wParam == VK_RIGHT) keyRight = false;
        if (wParam == VK_LEFT) keyLeft = false;
        if (wParam == VK_UP) keyUp = false;
        if (wParam == VK_DOWN) keyDown = false;
        return 0;
    
    case WM_PAINT:
        OnPain();
        return 0;

    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

HRESULT CreateResource()
{
    HRESULT hr = S_OK;
    if (rt == NULL)
    {
        D2D1_SIZE_U size = D2D1::SizeU(WIDTH, HEIGHT);
        hr = factory->CreateHwndRenderTarget
        (
            D2D1::RenderTargetProperties(),
            D2D1::HwndRenderTargetProperties(m_hwnd, size), &rt
        );

        if (SUCCEEDED(hr))
        {
            hr = rt->CreateSolidColorBrush(ColorF(ColorF::Black), &brush);
        }
    }

    return hr;
}

void DrawCircle(float x, float y, float r, ColorF col)
{
    D2D1_ELLIPSE ellipse = D2D1::Ellipse(D2D1::Point2F(x, y), r, r);
    brush->SetColor(ColorF(ColorF::Red));
    rt->FillEllipse(ellipse, brush);
}

float x = WIDTH / 2;
float y = HEIGHT / 2;

void OnPain()
{
    HRESULT hr = CreateResource();

    if (SUCCEEDED(hr))
    {
        PAINTSTRUCT ps;
        BeginPaint(m_hwnd, &ps);
        rt->BeginDraw();

        rt->Clear(ColorF(ColorF::AliceBlue));

        if (keyRight) x += 3;
        if (keyLeft) x -= 3;
        if (keyUp) y -= 3;
        if (keyDown) y += 3;

        DrawCircle(x, y, 30, ColorF(ColorF::Red));

        rt->EndDraw();

        EndPaint(m_hwnd, &ps);
        InvalidateRect(m_hwnd, NULL, FALSE);
    }
}
