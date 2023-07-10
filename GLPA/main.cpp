
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          //アプリケーションのインスタンスハンドル
    _In_opt_ HINSTANCE hPrevInstance,  //アプリケーション以前のインスタンスハンドルが入る。Win32アプリケーションでは常にNULL
    _In_ LPSTR lpCmdLine,              //コマンドラインが格納された、NULLで終わる文字列へのポインタが入る。
                                       //プログラム名は含まれない
    _In_ int nCmdShow                  //ウィンドウをどのように表示するかの指定が入る。SW_MESSAGENAMEの値が入る
)                      
{
    WNDCLASSEX wcex_LAU;

    wcex_LAU.cbSize = sizeof(wcex_LAU);                            //UINT WNDCLASSEX構造体の大きさの設定
    wcex_LAU.style = CS_HREDRAW | CS_VREDRAW;                      //UINT クラススタイルを表す。CS_MESSAGENAMEの値をO
                                                                   //R演算子で組み合わせた値となる
    wcex_LAU.lpfnWndProc = WndProc_LAU;                            //WNDPROC WNDPROCを指すポインタ
    wcex_LAU.cbClsExtra = 0;                                       //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex_LAU.cbWndExtra = 0;                                       //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex_LAU.hInstance = hInstance;                                //HINSTANCE インスタンスハンドル
    wcex_LAU.hIcon =                                               //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_LAU.hCursor =                                             //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex_LAU.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);           //HBRUSH クラス背景ブラシを指定する
    wcex_LAU.lpszMenuName = NULL;                                  //LPCSTR クラスメニューのリソース名を指定する
    wcex_LAU.lpszClassName = L"window_LAU";                        //LPCSTR ウィンドウクラスの名前を指定する
    wcex_LAU.hIconSm =                                             //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex_LAU.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex_LAU))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WNDCLASSEX wcex_PLAY;        //struct tagWNDCLASSEXW

    wcex_PLAY.cbSize = sizeof(wcex_PLAY);                                //UINT WNDCLASSEX構造体の大きさの設定
    wcex_PLAY.style = CS_HREDRAW | CS_VREDRAW;                           //UINT クラススタイルを表す。CS_MESSAGENAMEの値をOR演算子で組み合わせた値となる
    wcex_PLAY.lpfnWndProc = WndProc_PLAY;                                //WNDPROC WNDPROCを指すポインタ
    wcex_PLAY.cbClsExtra = 0;                                            //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex_PLAY.cbWndExtra = 0;                                            //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex_PLAY.hInstance = hInstance;                                     //HINSTANCE インスタンスハンドル
    wcex_PLAY.hIcon =                                                    //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_PLAY.hCursor =                                                  //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex_PLAY.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH クラス背景ブラシを指定する
    wcex_PLAY.lpszMenuName = NULL;                                       //LPCSTR クラスメニューのリソース名を指定する
    wcex_PLAY.lpszClassName = L"window_PLAY";                            //LPCSTR ウィンドウクラスの名前を指定する
    wcex_PLAY.hIconSm =                                                  //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex_PLAY.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex_PLAY))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window_PLAY"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WND_LAU.hWnd = CreateWindow(        //HWND ウィンドウハンドル
        L"window_LAU",                  //LPCSTR 登録されたクラス名のアドレス
        L"LAUNCHER",                    //LPCSTR ウィンドウテキストのアドレス
        WS_OVERLAPPEDWINDOW,            //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
        CW_USEDEFAULT, CW_USEDEFAULT,   //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
        WINDOW_WIDTH, WINDOW_HEIGHT,    //int ウィンドウの幅, ウィンドウの高さ
        HWND_DESKTOP,                   //HWND 親ウィンドウのハンドル
        NULL,                           //HMENU メニューのハンドルまたは子ウィンドウのID
        hInstance,                      //HINSTANCE アプリケーションインスタンスのハンドル
        NULL                            //void FAR* ウィンドウ作成データのアドレス
    );

    if (!WND_LAU.hWnd)
    {
        MessageBox(
            NULL,
            _T("window make fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    gr_hInstance = hInstance;
    gr_nCmdShow = nCmdShow;

    ShowWindow(
        WND_LAU.hWnd,
        nCmdShow
    );
    UpdateWindow(WND_LAU.hWnd);

    MSG msg;        //メッセージ構造体

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return (int)msg.wParam;             //関数がWM_QUITメッセージを受け取って終了したときは、メッセージのwParamパラメータが
							            //持つ終了コードを返す。関数がメッセージループに入る前に終了したときは、０を返す
}


LRESULT CALLBACK WndProc_LAU(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_KILLFOCUS:
            WND_LAU.foucus = false;
            return DefWindowProc(hWnd, message, wParam, lParam);


        case WM_SETFOCUS:
            WND_LAU.foucus = true;
            return DefWindowProc(hWnd, message, wParam, lParam);

        case WM_CREATE :
            {
                WND_LAU.hWndDC = GetDC(hWnd);
                WND_LAU.refreshRate = GetDeviceCaps(WND_LAU.hWndDC, VREFRESH);

                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / WND_LAU.refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                WND_LAU.hBufBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                WND_LAU.hBufBmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                WND_LAU.hBufBmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                WND_LAU.hBufBmpInfo.bmiHeader.biPlanes = 1;
                WND_LAU.hBufBmpInfo.bmiHeader.biBitCount = 32;
                WND_LAU.hBufBmpInfo.bmiHeader.biCompression = BI_RGB;
                
                WND_LAU.hBufDC = CreateCompatibleDC(WND_LAU.hWndDC);
                WND_LAU.hBufBmp = CreateDIBSection(NULL, &WND_LAU.hBufBmpInfo, DIB_RGB_COLORS, (LPVOID*)&WND_LAU.lpPixel, NULL, 0);
                SelectObject(WND_LAU.hBufDC, WND_LAU.hBufBmp);
                
                // //TODO:to make at texture.h,.cpp about load texture function
                // //load texture
                // sample.load(TEXT("sample.bmp"), WND_LAU.hWndDC);
                // sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_LAU.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                // sample.deleteImage(); 

                // sample2.load(TEXT("redimage.bmp"), WND_LAU.hWndDC);
                // sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_LAU.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                // sample2.deleteImage();   

                // sample3.load(TEXT("blueimage.bmp"), WND_LAU.hWndDC);
                // sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_LAU.hBufDC, WND_LAU.lpPixel);
                // texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                // sample3.deleteImage();     

                ReleaseDC(hWnd, WND_LAU.hWndDC);

                return 0;
            }

        case WM_CLOSE :
                DeleteDC(WND_LAU.hBufDC);
                // DeleteDC(hBmpDC);

                DeleteObject(WND_LAU.hBufBmp);
                // DeleteObject(hBmpFileBitmap);

                DestroyWindow(hWnd);

        case WM_DESTROY :
                //TODO:Change PostQuitMessage to send only when no windows are displayed.
                PostQuitMessage(0);

        case WM_PAINT :
                // OutputDebugString(L"debug window 1 drawing\n");
                WND_LAU.hWndDC = BeginPaint(hWnd, &WND_LAU.hPs);
                StretchDIBits(
                    WND_LAU.hWndDC,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    WND_LAU.lpPixel,
                    &WND_LAU.hBufBmpInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &WND_LAU.hPs);
                return 0;
                
        case WM_TIMER :
                switch (wParam)
                {
                    case REQUEST_ANIMATION_TIMER :
                            //fps
                            // OutputDebugString(L"debug window 1111111\n");
                            if (!WND_LAU.startFpsCount)
                            {
                                WND_LAU.lastLoopTime = clock();
                                WND_LAU.startFpsCount = true;
                            }
                            else
                            {
                                WND_LAU.thisLoopTime = clock();
                                WND_LAU.fps = 1000 / static_cast<long double>(WND_LAU.thisLoopTime - WND_LAU.lastLoopTime);
                                WND_LAU.fps = std::round(WND_LAU.fps * 100) / 100;
                                WND_LAU.lastLoopTime = WND_LAU.thisLoopTime;
                            }

                            PatBlt(
                                WND_LAU.hBufDC, 
                                0, 
                                0, 
                                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                                WHITENESS
                            );
                            scrLAUDwgContModif(WND_LAU.hBufDC);

                            InvalidateRect(hWnd, NULL, FALSE);
                            return 0;

                    case FPS_OUTPUT_TIMER :
                            _stprintf_s(mouseMsg, _T("FPS(%4.2lf)[fps]"), WND_LAU.fps);
                            return 0;
                            
                    default :
                            OutputDebugStringW(_T("TIMER ERROR\n"));
                            return 0;
                }
                
        case WM_KEYDOWN :
                if (!WND_LAU.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyDown(wParam);

                return 0;

        case WM_KEYUP :
                if (!WND_LAU.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyUp(wParam);
                return 0;

        case WM_LBUTTONDOWN :
                if (!WND_LAU.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.mouseLbtnDown(lParam);
                return 0;

        case WM_MOUSEMOVE :
                if (!WND_LAU.foucus)
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

LRESULT CALLBACK WndProc_PLAY(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_KILLFOCUS:
            WND_PLAY.foucus = false;
            return DefWindowProc(hWnd, message, wParam, lParam);


        case WM_SETFOCUS:
            WND_PLAY.foucus = true;
            return DefWindowProc(hWnd, message, wParam, lParam);

        case WM_CREATE :
            {
                WND_PLAY.hWndDC = GetDC(hWnd);
                WND_PLAY.refreshRate = GetDeviceCaps(WND_PLAY.hWndDC, VREFRESH);

                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / WND_PLAY.refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                WND_PLAY.hBufBmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                WND_PLAY.hBufBmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                WND_PLAY.hBufBmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                WND_PLAY.hBufBmpInfo.bmiHeader.biPlanes = 1;
                WND_PLAY.hBufBmpInfo.bmiHeader.biBitCount = 32;
                WND_PLAY.hBufBmpInfo.bmiHeader.biCompression = BI_RGB;
                
                WND_PLAY.hBufDC = CreateCompatibleDC(WND_PLAY.hWndDC);
                WND_PLAY.hBufBmp = CreateDIBSection(NULL, &WND_PLAY.hBufBmpInfo, DIB_RGB_COLORS, (LPVOID*)&WND_PLAY.lpPixel, NULL, 0);
                SelectObject(WND_PLAY.hBufDC, WND_PLAY.hBufBmp);
                
                // //TODO:to make at texture.h,.cpp about load texture function
                // //load texture
                // sample.load(TEXT("sample.bmp"), WND_PLAY.hWndDC);
                // sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_PLAY.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                // sample.deleteImage(); 

                // sample2.load(TEXT("redimage.bmp"), WND_PLAY.hWndDC);
                // sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_PLAY.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                // sample2.deleteImage();   

                // sample3.load(TEXT("blueimage.bmp"), WND_PLAY.hWndDC);
                // sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, WND_PLAY.hBufDC, WND_PLAY.lpPixel);
                // texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                // sample3.deleteImage();     

                ReleaseDC(hWnd, WND_PLAY.hWndDC);

                return 0;
            }

        case WM_CLOSE :
                DeleteDC(WND_PLAY.hBufDC);
                // DeleteDC(hBmpDC);

                DeleteObject(WND_PLAY.hBufBmp);
                // DeleteObject(hBmpFileBitmap);

                DestroyWindow(hWnd);

        case WM_DESTROY :
                //TODO:Change PostQuitMessage to send only when no windows are displayed.
                PostQuitMessage(0);

        case WM_PAINT :
                // OutputDebugString(L"debug window 1 drawing\n");
                WND_PLAY.hWndDC = BeginPaint(hWnd, &WND_PLAY.hPs);
                StretchDIBits(
                    WND_PLAY.hWndDC,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    WND_PLAY.lpPixel,
                    &WND_PLAY.hBufBmpInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &WND_PLAY.hPs);
                return 0;
                
        case WM_TIMER :
                switch (wParam)
                {
                    case REQUEST_ANIMATION_TIMER :
                            //fps
                            // OutputDebugString(L"debug window 1111111\n");
                            if (!WND_PLAY.startFpsCount)
                            {
                                WND_PLAY.lastLoopTime = clock();
                                WND_PLAY.startFpsCount = true;
                            }
                            else
                            {
                                WND_PLAY.thisLoopTime = clock();
                                WND_PLAY.fps = 1000 / static_cast<long double>(WND_PLAY.thisLoopTime - WND_PLAY.lastLoopTime);
                                WND_PLAY.fps = std::round(WND_PLAY.fps * 100) / 100;
                                WND_PLAY.lastLoopTime = WND_PLAY.thisLoopTime;
                            }

                            PatBlt(
                                WND_PLAY.hBufDC, 
                                0, 
                                0, 
                                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                                WHITENESS
                            );
                            scrPLAYDwgContModif(WND_PLAY.hBufDC);

                            InvalidateRect(hWnd, NULL, FALSE);
                            return 0;

                    case FPS_OUTPUT_TIMER :
                            _stprintf_s(mouseMsg, _T("FPS(%4.2lf)[fps]"), WND_PLAY.fps);
                            return 0;
                            
                    default :
                            OutputDebugStringW(_T("TIMER ERROR\n"));
                            return 0;
                }
                
        case WM_KEYDOWN :
                if (!WND_PLAY.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyDown(wParam);

                return 0;

        case WM_KEYUP :
                if (!WND_PLAY.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.keyUp(wParam);
                return 0;

        case WM_LBUTTONDOWN :
                if (!WND_PLAY.foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                UserInputWndLAU.mouseLbtnDown(lParam);
                return 0;

        case WM_MOUSEMOVE :
                if (!WND_PLAY.foucus)
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