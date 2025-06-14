
#include "userinput.h"

#include "window.h"

// WND LAU UserInput define

void WndLAUInput ::keyDown(WPARAM wParam)
{
        switch (wParam)
        {
        case VK_ESCAPE :
                _stprintf_s(szstr, _T("%s"), _T("ESCAPE"));
                // OutputDebugStringW(_T("ESCAPE\n"));
                break;
        case VK_SPACE :
                _stprintf_s(szstr, _T("%s"), _T("SPACE"));
                WndPLAY.hWnd = CreateWindow(           //HWND ウィンドウハンドル
                    L"window_PLAY",                 //LPCSTR 登録されたクラス名のアドレス
                    L"PLAY",                        //LPCSTR ウィンドウテキストのアドレス
                    WS_OVERLAPPEDWINDOW,            //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
                    CW_USEDEFAULT, CW_USEDEFAULT,   //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
                    WndPLAY.windowSize.width, WndPLAY.windowSize.height,    //int ウィンドウの幅, ウィンドウの高さ
                    HWND_DESKTOP,                   //HWND 親ウィンドウのハンドル
                    NULL,                           //HMENU メニューのハンドルまたは子ウィンドウのID
                    WndMain.hInstance,           //HINSTANCE アプリケーションインスタンスのハンドル
                    NULL                            //void FAR* ウィンドウ作成データのアドレス
                );

                if (!WndPLAY.hWnd)
                {
                        MessageBox(
                        NULL,
                        _T("window_PLAY make fail"),
                        _T("window_PLAY"),
                        MB_ICONEXCLAMATION
                        );

                    // return 1;
                }

                ShowWindow(
                        WndPLAY.hWnd,
                        WndMain.nCmdShow
                );

                UpdateWindow(WndPLAY.hWnd);
                // OutputDebugStringW(_T("SPACE\n"));
                break;
        case VK_SHIFT :
                _stprintf_s(szstr, _T("%s"), _T("SHIFT"));
                // OutputDebugStringW(_T("SHIFT\n"));
                break;
        case 'W' :
                _stprintf_s(szstr, _T("%s"), _T("W ON"));
                WndLAU.fpsSystem.setFps = 240;
                // OutputDebugStringW(_T("W\n"));
                break;
        case 'S' :
                _stprintf_s(szstr, _T("%s"), _T("S ON"));
                WndLAU.fpsSystem.setFps = 10;
                break;
        case 'C' :
                mainCam.initialize();
                mainCam.defClippingArea();
                mainCam.coordinateTransRange(&tempObject.data);

                break;
        default :
                _stprintf_s(szstr, _T("%s"), _T("ANY"));
                break;
        }
};

void WndLAUInput ::keyUp(WPARAM wParam)
{
        switch (wParam)
        {
                case VK_ESCAPE :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("ESCAPE UP\n"));
                        break;
                case VK_SPACE :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SPACE UP\n"));
                        break;
                case VK_SHIFT :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SHIFT UP\n"));
                        break;
                case 'W' :
                        _stprintf_s(szstr, _T("%s"), _T("W OFF"));
                        // OutputDebugStringW(_T("W UP\n"));
                        break;
                case 'S' :
                        _stprintf_s(szstr, _T("%s"), _T("S OFF"));
                        break;
                default :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        break;
        }
};

void WndLAUInput :: mouseMove(LPARAM lParam)
{
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};


void WndLAUInput :: mouseLbtnDown(LPARAM lParam)
{
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};

// WND Play UserInput define
void WndPLAYInput ::keyDown(WPARAM wParam)
{
        switch (wParam)
        {
        case VK_ESCAPE :
                _stprintf_s(szstrfPlay, _T("%s"), _T("ESCAPE"));
                // OutputDebugStringW(_T("ESCAPE\n"));
                break;
        case VK_SPACE :
                _stprintf_s(szstrfPlay, _T("%s"), _T("SPACE"));
        
                break;
        case VK_SHIFT :
                _stprintf_s(szstrfPlay, _T("%s"), _T("SHIFT"));
                // OutputDebugStringW(_T("SHIFT\n"));
                break;
        case 'W' :
                _stprintf_s(szstrfPlay, _T("%s"), _T("W ON"));
                // OutputDebugStringW(_T("W\n"));
                break;
        default :
                _stprintf_s(szstrfPlay, _T("%s"), _T("ANY"));
                break;
        }
};

void WndPLAYInput ::keyUp(WPARAM wParam)
{
        switch (wParam)
        {
                case VK_ESCAPE :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("ESCAPE UP\n"));
                        break;
                case VK_SPACE :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SPACE UP\n"));
                        break;
                case VK_SHIFT :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SHIFT UP\n"));
                        break;
                case 'W' :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("W OFF"));
                        // OutputDebugStringW(_T("W UP\n"));
                        break;
                default :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        break;
        }
};

void WndPLAYInput :: mouseMove(LPARAM lParam)
{
        ptfPlay.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        ptfPlay.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};


void WndPLAYInput :: mouseLbtnDown(LPARAM lParam)
{
        ptfPlay.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
        ptfPlay.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        _stprintf_s(szstrfPlay, _T("%d,%d"), ptfPlay.x, ptfPlay.y);
};


WndLAUInput UserInputWndLAU;
WndPLAYInput UserInputWndPLAY;
