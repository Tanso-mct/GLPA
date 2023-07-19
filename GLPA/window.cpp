
#include "window.h"

WNDMAIN WndMain;
WINDOW_LAU WndLAU;
WINDOW_PLAY WndPLAY;

WNDCLASSEX WNDMAIN::registerClass
(
    UINT style, 
    WNDPROC wndproc, 
    int clsExtra, 
    int wndExtra, 
    HINSTANCE hInstance, 
    LPWSTR loadIcon, 
    LPWSTR loadCursor, 
    int backgroundColor, 
    LPCWSTR menuResName, 
    LPCWSTR name, 
    LPWSTR smallIcon
)
{
    WNDCLASSEX wndClass;
    wndClass.cbSize = sizeof(wndClass);     //UINT WNDCLASSEX構造体の大きさの設定
    wndClass.style = style;                 //UINT クラススタイルを表す。CS_MESSAGENAMEの値をOR演算子で組み合わせた値となる
    wndClass.lpfnWndProc = wndproc;         //WNDPROC WNDPROCを指すポインタ
    wndClass.cbClsExtra = clsExtra;         //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wndClass.cbWndExtra = wndExtra;         //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wndClass.hInstance = hInstance;         //HINSTANCE インスタンスハンドル
    wndClass.hIcon = (HICON)LoadImage       //HICON クラスアイコンを指定するLoadImage
        (
            NULL, 
            MAKEINTRESOURCE(loadIcon),
            IMAGE_ICON,
            0,
            0,
            LR_DEFAULTSIZE | LR_SHARED
        );
    wndClass.hCursor = (HCURSOR)LoadImage   //HCURSOR クラスカーソルを指定する
        (
            NULL, 
            MAKEINTRESOURCE(loadIcon),
            IMAGE_CURSOR,
            0,
            0,
            LR_DEFAULTSIZE | LR_SHARED
        );                                                 
    wndClass.hbrBackground = (HBRUSH)GetStockObject(backgroundColor);       //HBRUSH クラス背景ブラシを指定する
    wndClass.lpszMenuName = menuResName;                                    //LPCSTR クラスメニューのリソース名を指定する
    wndClass.lpszClassName = name;                                          //LPCSTR ウィンドウクラスの名前を指定する
    wndClass.hIconSm =                                                      //HICON 小さなクラスアイコンを指定する
    LoadIcon(wndClass.hInstance, MAKEINTRESOURCE(smallIcon));
    return wndClass;
}

int WNDMAIN::checkClass(WNDCLASSEX *ptClass)
{
    if (!RegisterClassEx(ptClass))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return NULL;
    }
}

int WNDMAIN::checkWindow(HWND createdHWnd)
{
    if (!createdHWnd)
    {
        MessageBox(
            NULL,
            _T("window make fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return NULL;
    }
}


LRESULT CALLBACK WINDOW_LAU::wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_KILLFOCUS:
            WndLAU.state.foucus = false;
            OutputDebugString(L"WND_LAU K\n");
            return 0;


        case WM_SETFOCUS:
            WndLAU.state.foucus = true;
            OutputDebugString(L"WND_LAU S\n");
            return 0;

        case WM_CREATE :
            {
                WndLAU.state.open = true;
                WndLAU.hWndDC = GetDC(hWnd);
                WndLAU.fps.refreshRate = GetDeviceCaps(WndLAU.hWndDC, VREFRESH);

                // TODO: Use TIMERPROC function with setTimer
                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, (UINT)std::floor(1000 / WndLAU.fps.refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biPlanes = 1;
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biBitCount = 32;
                WndLAU.buffer.hBufBmpInfo.bmiHeader.biCompression = BI_RGB;
                
                WndLAU.buffer.hBufDC = CreateCompatibleDC(WndLAU.hWndDC);
                WndLAU.buffer.hBufBmp = CreateDIBSection
                (
                    NULL, 
                    &WndLAU.buffer.hBufBmpInfo, 
                    DIB_RGB_COLORS, 
                    (LPVOID*)&WndLAU.buffer.lpPixel, 
                    NULL, 
                    0
                );
                SelectObject(WndLAU.buffer.hBufDC, WndLAU.buffer.hBufBmp);
                
                // //TODO:to make at texture.h,.cpp about load texture function
                // //load texture
                // sample.load(TEXT("sample.bmp"), WND_LAU.hWndDC);
                // sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndLAU.buffer.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                // sample.deleteImage(); 

                // sample2.load(TEXT("redimage.bmp"), WND_LAU.hWndDC);
                // sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndLAU.buffer.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                // sample2.deleteImage();   

                // sample3.load(TEXT("blueimage.bmp"), WND_LAU.hWndDC);
                // sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndLAU.buffer.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                // sample3.deleteImage();     

                ReleaseDC(hWnd, WndLAU.hWndDC);

                return 0;
            }

        case WM_CLOSE :
                WndLAU.state.open = false;
                DeleteDC(WndLAU.buffer.hBufDC);
                // DeleteDC(hBmpDC);

                DeleteObject(WndLAU.buffer.hBufBmp);
                // DeleteObject(hBmpFileBitmap);

                DestroyWindow(hWnd);

        case WM_DESTROY :
                //TODO:Change PostQuitMessage to send only when no windows are displayed.
                if (!WndPLAY.state.open)
                {
                    PostQuitMessage(0);
                }

                return 0;

        case WM_PAINT :
                // OutputDebugString(L"debug window 1 drawing\n");
                WndLAU.hWndDC = BeginPaint(hWnd, &WndLAU.hPs);
                StretchDIBits(
                    WndLAU.hWndDC,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    WndLAU.buffer.lpPixel,
                    &WndLAU.buffer.hBufBmpInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &WndLAU.hPs);
                return 0;
                
        case WM_TIMER :
                switch (wParam)
                {
                    case REQUEST_ANIMATION_TIMER :
                            //fps
                            // OutputDebugString(L"debug window 1111111\n");
                            if (!WndLAU.fps.startFpsCount)
                            {
                                WndLAU.fps.lastLoopTime = clock();
                                WndLAU.fps.startFpsCount = true;
                            }
                            else
                            {
                                WndLAU.fps.thisLoopTime = clock();
                                WndLAU.fps.fps = 1000 / static_cast<long double>(WndLAU.fps.thisLoopTime - WndLAU.fps.lastLoopTime);
                                WndLAU.fps.fps = std::round(WndLAU.fps.fps * 100) / 100;
                                WndLAU.fps.lastLoopTime = WndLAU.fps.thisLoopTime;
                            }

                            PatBlt(
                                WndLAU.buffer.hBufDC, 
                                0, 
                                0, 
                                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                                WHITENESS
                            );
                            scrLAUDwgContModif(WndLAU.buffer.hBufDC);

                            InvalidateRect(hWnd, NULL, FALSE);
                            return 0;

                    case FPS_OUTPUT_TIMER :
                            _stprintf_s(mouseMsg, _T("FPS(%4.2lf)[fps]"), WndLAU.fps.fps);
                            return 0;
                            
                    default :
                            OutputDebugStringW(_T("TIMER ERROR\n"));
                            return 0;
                }
                
        case WM_KEYDOWN :
                if (!WndLAU.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyDown(wParam);

                return 0;

        case WM_KEYUP :
                if (!WndLAU.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyUp(wParam);
                return 0;

        case WM_LBUTTONDOWN :
                if (!WndLAU.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.mouseLbtnDown(lParam);
                return 0;

        case WM_MOUSEMOVE :
                if (!WndLAU.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.mouseMove(lParam);
                return 0;
            
        default :
                return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

LRESULT CALLBACK WINDOW_PLAY::wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_KILLFOCUS:
            WndPLAY.state.foucus = false;
            OutputDebugString(L"WND_PLAY K\n");

            return 0;

        case WM_SETFOCUS:
            WndPLAY.state.foucus = true;
            OutputDebugString(L"WND_PLAY S\n");

            return 0;   

        case WM_CREATE :
            {
                WndPLAY.state.open = true;
                WndPLAY.hWndDC = GetDC(hWnd);
                WndPLAY.fps.refreshRate = GetDeviceCaps(WndPLAY.hWndDC, VREFRESH);

                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, (UINT)std::floor(1000 / WndPLAY.fps.refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biPlanes = 1;
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biBitCount = 32;
                WndPLAY.buffer.hBufBmpInfo.bmiHeader.biCompression = BI_RGB;
                
                WndPLAY.buffer.hBufDC = CreateCompatibleDC(WndPLAY.hWndDC);
                WndPLAY.buffer.hBufBmp = CreateDIBSection
                (
                    NULL, 
                    &WndPLAY.buffer.hBufBmpInfo, 
                    DIB_RGB_COLORS, 
                    (LPVOID*)&WndPLAY.buffer.lpPixel, 
                    NULL, 
                    0
                );
                SelectObject(WndPLAY.buffer.hBufDC, WndPLAY.buffer.hBufBmp);
                
                // //TODO:to make at texture.h,.cpp about load texture function
                // //load texture
                // sample.load(TEXT("sample.bmp"), WndPLAY.hWndDC);
                // sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndPLAY.buffer.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                // sample.deleteImage(); 

                // sample2.load(TEXT("redimage.bmp"), WndPLAY.hWndDC);
                // sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndPLAY.buffer.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                // sample2.deleteImage();   

                // sample3.load(TEXT("blueimage.bmp"), WndPLAY.hWndDC);
                // sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WndPLAY.buffer.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                // sample3.deleteImage();     

                ReleaseDC(hWnd, WndPLAY.hWndDC);

                return 0;
            }

        case WM_CLOSE :
                WndPLAY.state.open = false;
                DeleteDC(WndPLAY.buffer.hBufDC);
                // DeleteDC(hBmpDC);

                DeleteObject(WndPLAY.buffer.hBufBmp);
                // DeleteObject(hBmpFileBitmap);

                DestroyWindow(hWnd);

        case WM_DESTROY :
                //TODO:Change PostQuitMessage to send only when no windows are displayed.
                if (!WndLAU.state.open)
                {
                    PostQuitMessage(0);
                }

                return 0;

        case WM_PAINT :
                // OutputDebugString(L"debug window 1 drawing\n");
                WndPLAY.hWndDC = BeginPaint(hWnd, &WndPLAY.hPs);
                StretchDIBits(
                    WndPLAY.hWndDC,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    WndPLAY.buffer.lpPixel,
                    &WndPLAY.buffer.hBufBmpInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &WndPLAY.hPs);
                return 0;
                
        case WM_TIMER :
                switch (wParam)
                {
                    case REQUEST_ANIMATION_TIMER :
                            //fps
                            // OutputDebugString(L"debug window 1111111\n");
                            if (!WndPLAY.fps.startFpsCount)
                            {
                                WndPLAY.fps.lastLoopTime = clock();
                                WndPLAY.fps.startFpsCount = true;
                            }
                            else
                            {
                                WndPLAY.fps.thisLoopTime = clock();
                                WndPLAY.fps.fps 
                                = 1000 / static_cast<long double>(WndPLAY.fps.thisLoopTime - WndPLAY.fps.lastLoopTime);
                                WndPLAY.fps.fps = std::round(WndPLAY.fps.fps * 100) / 100;
                                WndPLAY.fps.lastLoopTime = WndPLAY.fps.thisLoopTime;
                            }

                            PatBlt(
                                WndPLAY.buffer.hBufDC, 
                                0, 
                                0, 
                                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                                WHITENESS
                            );
                            scrPLAYDwgContModif(WndPLAY.buffer.hBufDC);

                            InvalidateRect(hWnd, NULL, FALSE);
                            return 0;

                    case FPS_OUTPUT_TIMER :
                            _stprintf_s(mouseMsgfPlay, _T("FPS(%4.2lf)[fps]"), WndPLAY.fps.fps);
                            return 0;
                            
                    default :
                            OutputDebugStringW(_T("TIMER ERROR\n"));
                            return 0;
                }
                
        case WM_KEYDOWN :
                if (!WndPLAY.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndPLAY.keyDown(wParam);

                return 0;

        case WM_KEYUP :
                if (!WndPLAY.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndPLAY.keyUp(wParam);
                return 0;

        case WM_LBUTTONDOWN :
                if (!WndPLAY.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndPLAY.mouseLbtnDown(lParam);
                return 0;

        case WM_MOUSEMOVE :
                if (!WndPLAY.state.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndPLAY.mouseMove(lParam);
                return 0;
            
        default :
                return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
