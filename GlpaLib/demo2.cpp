#include <windows.h>
#include <d2d1_1.h> // 追加
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using namespace Microsoft::WRL;

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    // ウィンドウクラスの登録
    WNDCLASSEX wcex = {};
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.lpszClassName = L"Direct2DWindowClass";
    RegisterClassEx(&wcex);

    // ウィンドウの作成
    HWND hwnd = CreateWindow(L"Direct2DWindowClass", L"Direct2D and Swap Chain Example", WS_OVERLAPPEDWINDOW, 
                             CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, nullptr, nullptr, hInstance, nullptr);

    ShowWindow(hwnd, nCmdShow);

    // Direct3D デバイスとスワップチェインの作成
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    ComPtr<ID3D11Device> d3dDevice;
    ComPtr<ID3D11DeviceContext> d3dContext;
    ComPtr<IDXGISwapChain1> swapChain;
    D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, featureLevels, 1,
                      D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dContext);

    ComPtr<IDXGIDevice> dxgiDevice;
    d3dDevice.As(&dxgiDevice);

    ComPtr<IDXGIAdapter> dxgiAdapter;
    dxgiDevice->GetAdapter(&dxgiAdapter);

    ComPtr<IDXGIFactory2> dxgiFactory;
    dxgiAdapter->GetParent(__uuidof(IDXGIFactory2), &dxgiFactory);

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = 2;
    swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapChainDesc.Width = 800;
    swapChainDesc.Height = 600;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

    dxgiFactory->CreateSwapChainForHwnd(d3dDevice.Get(), hwnd, &swapChainDesc, nullptr, nullptr, &swapChain);

    // Direct2D デバイスとコンテキストの作成
    ComPtr<ID2D1Factory1> d2dFactory;
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2dFactory.GetAddressOf());

    ComPtr<ID2D1Device> d2dDevice;
    d2dFactory->CreateDevice(dxgiDevice.Get(), &d2dDevice);

    ComPtr<ID2D1DeviceContext> d2dContext;
    d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &d2dContext);

    // スワップチェインのバックバッファからビットマップを作成
    ComPtr<IDXGISurface> dxgiBackBuffer;
    swapChain->GetBuffer(0, __uuidof(IDXGISurface), &dxgiBackBuffer);

    D2D1_BITMAP_PROPERTIES1 bitmapProperties = {};
    bitmapProperties.pixelFormat.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bitmapProperties.pixelFormat.alphaMode = D2D1_ALPHA_MODE_IGNORE;

    ComPtr<ID2D1Bitmap1> d2dBitmap;
    d2dContext->CreateBitmapFromDxgiSurface(dxgiBackBuffer.Get(), &bitmapProperties, &d2dBitmap);

    // メインループ
    MSG msg = {};
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            // 描画開始
            d2dContext->SetTarget(d2dBitmap.Get());
            d2dContext->BeginDraw();

            d2dContext->Clear(D2D1::ColorF(D2D1::ColorF::White));

            // 四角形を描画
            D2D1_RECT_F rectangle = D2D1::RectF(100.0f, 100.0f, 300.0f, 300.0f);
            ComPtr<ID2D1SolidColorBrush> brush;
            d2dContext->CreateSolidColorBrush(D2D1::ColorF(D2D1::ColorF::Red), &brush);
            d2dContext->FillRectangle(&rectangle, brush.Get());

            d2dContext->EndDraw();

            // スワップチェインのバックバッファを表示
            swapChain->Present(1, 0);
        }
    }

    return static_cast<int>(msg.wParam);
}
