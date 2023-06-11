
#include "main.h"
#include "bmp.h"
#include "graphic.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,           //アプリケーションのインスタンスハンドル
    _In_opt_ HINSTANCE hPrevInstance,   //アプリケーション以前のインスタンスハンドルが入る。Win32アプリケーションでは常にNULL
    _In_ LPSTR lpCmdLine,               //コマンドラインが格納された、NULLで終わる文字列へのポインタが入る。プログラム名は含まれない
    _In_ int nCmdShow)                  //ウィンドウをどのように表示するかの指定が入る。SW_MESSAGENAMEの値が入る  
{
    WNDCLASSEX wcex;        //struct tagWNDCLASSEXW

    wcex.cbSize = sizeof(wcex);                                     //UINT WNDCLASSEX構造体の大きさの設定
    wcex.style = CS_HREDRAW | CS_VREDRAW;                           //UINT クラススタイルを表す。CS_MESSAGENAMEの値をOR演算子で組み合わせた値となる
    wcex.lpfnWndProc = WndProc;                                     //WNDPROC WNDPROCを指すポインタ
    wcex.cbClsExtra = 0;                                            //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex.cbWndExtra = 0;                                            //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex.hInstance = hInstance;                                     //HINSTANCE インスタンスハンドル
    wcex.hIcon =                                                    //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex.hCursor =                                                  //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH クラス背景ブラシを指定する
    wcex.lpszMenuName = NULL;                                       //LPCSTR クラスメニューのリソース名を指定する
    wcex.lpszClassName = L"window1";                                //LPCSTR ウィンドウクラスの名前を指定する
    wcex.hIconSm =                                                  //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window1"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WNDCLASSEX wcex2;        //struct tagWNDCLASSEXW

    wcex2.cbSize = sizeof(wcex2);                                    //UINT WNDCLASSEX構造体の大きさの設定
    wcex2.style = CS_HREDRAW | CS_VREDRAW;                           //UINT クラススタイルを表す。CS_MESSAGENAMEの値をOR演算子で組み合わせた値となる
    wcex2.lpfnWndProc = WndProc2;                                    //WNDPROC WNDPROCを指すポインタ
    wcex2.cbClsExtra = 0;                                            //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex2.cbWndExtra = 0;                                            //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex2.hInstance = hInstance;                                     //HINSTANCE インスタンスハンドル
    wcex2.hIcon =                                                    //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex2.hCursor =                                                  //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex2.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH クラス背景ブラシを指定する
    wcex2.lpszMenuName = NULL;                                       //LPCSTR クラスメニューのリソース名を指定する
    wcex2.lpszClassName = L"window2";                                //LPCSTR ウィンドウクラスの名前を指定する
    wcex2.hIconSm =                                                  //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex2.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex2))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window2"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    HWND hWnd = CreateWindow(           //HWND ウィンドウハンドル
        L"window1",                     //LPCSTR 登録されたクラス名のアドレス
        L"GLPA",                        //LPCSTR ウィンドウテキストのアドレス
        WS_OVERLAPPEDWINDOW,            //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
        CW_USEDEFAULT, CW_USEDEFAULT,   //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
        WINDOW_WIDTH, WINDOW_HEIGHT,    //int ウィンドウの幅, ウィンドウの高さ
        HWND_DESKTOP,                   //HWND 親ウィンドウのハンドル
        NULL,                           //HMENU メニューのハンドルまたは子ウィンドウのID
        hInstance,                      //HINSTANCE アプリケーションインスタンスのハンドル
        NULL                            //void FAR* ウィンドウ作成データのアドレス
    );

    if (!hWnd)
    {
        MessageBox(
            NULL,
            _T("window make fail"),
            _T("window1"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    gr_hInstance = hInstance;

    gr_nCmdShow = nCmdShow;

    ShowWindow(
        hWnd,
        nCmdShow
    );
    UpdateWindow(hWnd);

    MSG msg;        //メッセージ構造体

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return (int)msg.wParam;             //関数がWM_QUITメッセージを受け取って終了したときは、メッセージのwParamパラメータが
							            //持つ終了コードを返す。関数がメッセージループに入る前に終了したときは、０を返す
}

void draw(HDC hBuffer_DC, TEXTURE *texture)
{
    texture->displayImage_rectangle(
        lpPixel, texture->file1.bmp_pixel, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, 
        0, 0,
        0, 0,
        FILE_MAXPIXEL_X, FILE_MAXPIXEL_Y
    );

    // texture->displayImage_rectangle(
    //     lpPixel, texture->file2.bmp_pixel, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, 
    //     0, 0,
    //     0, 0,
    //     800, 800
    // );

    HFONT hFont1 = CreateFont(30 * DISPLAY_RESOLUTION, 0, 
		0, 0, 0, 
		FALSE, FALSE, FALSE,   
		SHIFTJIS_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,  
		VARIABLE_PITCH | FF_ROMAN, NULL);   
	SelectObject(hBuffer_DC, hFont1);  
    
    TextOut(
        hBuffer_DC,
        pt.x,
        pt.y,
        mouseMsg,
        _tcslen(mouseMsg)
    );
    DeleteObject(hFont1); 

    TextOut(
        hBuffer_DC,
        5,
        5,
        szstr,
        _tcslen(szstr)
    );
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    // if (hWnd2Open && message != WM_SETFOCUS)
    // {
    //     return DefWindowProc(hWnd, message, wParam, lParam);
    // }
    // else if (message == WM_SETFOCUS)
    // {
    //     hWnd1Open = true;
    //     hWnd2Open = false;
    // }


    switch (message)
    {
        case WM_KILLFOCUS:
            hWnd1_foucus = false;
            return DefWindowProc(hWnd, message, wParam, lParam);


        case WM_SETFOCUS:
            hWnd1_foucus = true;
            return DefWindowProc(hWnd, message, wParam, lParam);

        case WM_CREATE :
            {
                hWindow_DC = GetDC(hWnd);
                refreshRate = GetDeviceCaps(hWindow_DC, VREFRESH);

                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                hBuffer_bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                hBuffer_bitmapInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                hBuffer_bitmapInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                hBuffer_bitmapInfo.bmiHeader.biPlanes = 1;
                hBuffer_bitmapInfo.bmiHeader.biBitCount = 32;
                hBuffer_bitmapInfo.bmiHeader.biCompression = BI_RGB;
                
                hBuffer_DC = CreateCompatibleDC(hWindow_DC);
                hBuffer_bitmap = CreateDIBSection(NULL, &hBuffer_bitmapInfo, DIB_RGB_COLORS, (LPVOID*)&lpPixel, NULL, 0);
                SelectObject(hBuffer_DC, hBuffer_bitmap);
                // SelectObject(hBuffer_DC, GetStockObject(DC_PEN));
                // SelectObject(hBuffer_DC, GetStockObject(DC_BRUSH)); 
                
                //bmp file dc
                // bmpFileInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                // bmpFileInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                // bmpFileInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                // bmpFileInfo.bmiHeader.biPlanes = 1;
                // bmpFileInfo.bmiHeader.biBitCount = 32;
                // bmpFileInfo.bmiHeader.biCompression = BI_RGB;
                
                // hBmpDC = CreateCompatibleDC(hWindow_DC);
                // hBmpFileBitmap = CreateDIBSection(NULL, &bmpFileInfo, DIB_RGB_COLORS, (LPVOID*)&bmpPixel, NULL, 0);
                // SelectObject(hBmpDC, hBmpFileBitmap);
                
                //load texture
                sample.load(TEXT("sample.bmp"), hWindow_DC);
                sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBuffer_DC, lpPixel);
                texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                sample.deleteImage(); 

                sample2.load(TEXT("redimage.bmp"), hWindow_DC);
                sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBuffer_DC, lpPixel);
                texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                sample2.deleteImage();   

                sample3.load(TEXT("blueimage.bmp"), hWindow_DC);
                sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBuffer_DC, lpPixel);
                texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                sample3.deleteImage();     

                // DeleteDC(hBmpDC);
                ReleaseDC(hWnd, hWindow_DC);

                //full screen
                // SetMenu(hWnd, NULL);
                // SetWindowLong(hWnd, GWL_STYLE, WS_VISIBLE | WS_BORDER);
                // MoveWindow(
                //     hWnd,
                //     0, 0,
                //     WINDOW_WIDTH, WINDOW_HEIGHT,
                //     FALSE
                // );
                return 0;
            }

        case WM_ERASEBKGND :
                return 1;
        case WM_PAINT :
                OutputDebugString(L"debug window 1 drawing\n");
                hWindow_DC = BeginPaint(hWnd, &hPS);
                StretchDIBits(
                    hWindow_DC,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    lpPixel,
                    &hBuffer_bitmapInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &hPS);
                return 0;
                
        case WM_TIMER :
            switch (wParam)
            {
                case REQUEST_ANIMATION_TIMER :
                        //fps
                        // OutputDebugString(L"debug window 1111111\n");
                        if (!startFpsCount)
                        {
                            lastloop = clock();
                            startFpsCount = true;
                        }
                        else
                        {
                            thisloop = clock();
                            fps = 1000 / static_cast<long double>(thisloop - lastloop);
                            fps = std::round(fps * 100) / 100;
                            lastloop = thisloop;
                        }

                        PatBlt(
                            hBuffer_DC, 
                            0, 
                            0, 
                            WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                            WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                            WHITENESS
                        );
                        draw(hBuffer_DC, pt_texture_sample);

                        InvalidateRect(hWnd, NULL, FALSE);
                        return 0;
                case FPS_OUTPUT_TIMER :
                        _stprintf_s(mouseMsg, _T("FPS(%4.2lf)[fps]"), fps);
                        return 0;
                default :
                        OutputDebugStringW(_T("TIMER ERROR\n"));
                        return 0;
            }
                
        case WM_KEYDOWN :
                if (!hWnd1_foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                switch (wParam)
                {
                    case VK_ESCAPE :
                            _stprintf_s(szstr, _T("%s"), _T("ESCAPE"));
                            // OutputDebugStringW(_T("ESCAPE\n"));
                            break;
                    case VK_SPACE :
                            _stprintf_s(szstr, _T("%s"), _T("SPACE"));
                            //full screen
                            hWnd1Open = false;
                            hWnd2 = CreateWindow(           //HWND ウィンドウハンドル
                                L"window2",                     //LPCSTR 登録されたクラス名のアドレス
                                L"GLPA2",                        //LPCSTR ウィンドウテキストのアドレス
                                WS_OVERLAPPEDWINDOW,            //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
                                CW_USEDEFAULT, CW_USEDEFAULT,   //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
                                WINDOW_WIDTH, WINDOW_HEIGHT,    //int ウィンドウの幅, ウィンドウの高さ
                                HWND_DESKTOP,                           //HWND 親ウィンドウのハンドル
                                NULL,                           //HMENU メニューのハンドルまたは子ウィンドウのID
                                gr_hInstance,                      //HINSTANCE アプリケーションインスタンスのハンドル
                                NULL                            //void FAR* ウィンドウ作成データのアドレス
                            );

                            if (!hWnd2)
                            {
                                MessageBox(
                                    NULL,
                                    _T("window2 make fail"),
                                    _T("window2"),
                                    MB_ICONEXCLAMATION
                                );

                                return 1;
                            }
                            // else
                            // {
                            //     hWnd2Open = true;
                            // }

                            ShowWindow(
                                hWnd2,
                                gr_nCmdShow
                            );

                            UpdateWindow(gr_hWnd2);
                            // SetMenu(hWnd, NULL);
                            // SetWindowLong(hWnd, GWL_STYLE, WS_VISIBLE | WS_BORDER);
                            // MoveWindow(
                            //     hWnd,
                            //     0, 0,
                            //     WINDOW_WIDTH, WINDOW_HEIGHT,
                            //     FALSE
                            // );
                            // OutputDebugStringW(_T("SPACE\n"));
                            break;
                    case VK_SHIFT :
                            _stprintf_s(szstr, _T("%s"), _T("SHIFT"));
                            // OutputDebugStringW(_T("SHIFT\n"));
                            break;
                    case 'W' :
                            _stprintf_s(szstr, _T("%s"), _T("W ON"));
                            // OutputDebugStringW(_T("W\n"));
                            break;
                    default :
                            _stprintf_s(szstr, _T("%s"), _T("ANY"));
                            break;
                }
                return 0;

        case WM_KEYUP :
                if (!hWnd1_foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

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
                    default :
                            _stprintf_s(szstr, _T("%s"), _T("NAN"));
                            break;
                }
                return 0;

        case WM_LBUTTONDOWN :
                if (!hWnd1_foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }

                pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
                pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
                // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
                return 0;

        case WM_MOUSEMOVE :
                if (!hWnd1_foucus)
                {
                    return DefWindowProc(hWnd, message, wParam, lParam);
                }
                
                pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
                pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
                // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
                return 0;

        case WM_CLOSE :
                DeleteDC(hBuffer_DC);
                // DeleteDC(hBmpDC);

                DeleteObject(hBuffer_bitmap);
                // DeleteObject(hBmpFileBitmap);

                DestroyWindow(hWnd);

        case WM_DESTROY :
                PostQuitMessage(0);
            
        default :
                return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

LRESULT CALLBACK WndProc2(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    //win32 define
    PAINTSTRUCT hPS;
    HDC hWindow_DC;

    //mouse move limit
    static RECT rc =
    {
        0,
        0,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    };
    static int debug_num = 0;
    if (debug_num == 0)
    {
        ClipCursor(&rc);
        debug_num = 1;
    }
    

    //bufer bmp dc
    static HBITMAP hBuffer_bitmap;
    static HDC hBuffer_DC;
    static BITMAPINFO hBuffer_bitmapInfo;

    //bmpfile dc
    static HBITMAP hBmpFileBitmap;
    static HDC hBmpDC;
    static BITMAPINFO bmpFileInfo;

    //textute
    static TEXTURE texture_sample;
    static TEXTURE* pt_texture_sample = &texture_sample;

    //bmpfile
    static BMPFILE sample;
    static BMPFILE* pt_sample = &sample;

    static BMPFILE sample2;
    static BMPFILE* pt_sample2 = &sample2;

    static BMPFILE sample3;
    static BMPFILE* pt_sample3 = &sample3;

    //fps
    static int refreshRate;
    static bool startFpsCount = false;
    static clock_t thisloop;
    static clock_t lastloop;
    static long double fps;

    if (hWnd1Open && message != WM_SETFOCUS)
    {
        return DefWindowProc(hWnd, message, wParam, lParam);;
    }
    else if (message == WM_SETFOCUS)
    {
        hWnd1Open = false;
        hWnd2Open = true;
    }

    switch (message)
    {
    case WM_CREATE:
    {
        hWindow_DC = GetDC(hWnd);
        refreshRate = GetDeviceCaps(hWindow_DC, VREFRESH);
        ReleaseDC(hWnd, hWindow_DC);

        SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / refreshRate), NULL);
        SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);

        //bmp buffer dc
        hBuffer_bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        hBuffer_bitmapInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
        hBuffer_bitmapInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;
        hBuffer_bitmapInfo.bmiHeader.biPlanes = 1;
        hBuffer_bitmapInfo.bmiHeader.biBitCount = 32;
        hBuffer_bitmapInfo.bmiHeader.biCompression = BI_RGB;

        hWindow_DC = GetDC(hWnd);
        hBuffer_DC = CreateCompatibleDC(hWindow_DC);
        hBuffer_bitmap = CreateDIBSection(NULL, &hBuffer_bitmapInfo, DIB_RGB_COLORS, (LPVOID*)&lpPixel, NULL, 0);
        SelectObject(hBuffer_DC, hBuffer_bitmap);
        SelectObject(hBuffer_DC, GetStockObject(DC_PEN));
        SelectObject(hBuffer_DC, GetStockObject(DC_BRUSH));

        //bmp file dc
        bmpFileInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmpFileInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
        bmpFileInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;
        bmpFileInfo.bmiHeader.biPlanes = 1;
        bmpFileInfo.bmiHeader.biBitCount = 32;
        bmpFileInfo.bmiHeader.biCompression = BI_RGB;

        hBmpDC = CreateCompatibleDC(hWindow_DC);
        hBmpFileBitmap = CreateDIBSection(NULL, &bmpFileInfo, DIB_RGB_COLORS, (LPVOID*)&bmpPixel, NULL, 0);
        SelectObject(hBmpDC, hBmpFileBitmap);

        //load texture
        sample.load(TEXT("sample.bmp"), hWindow_DC);
        sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
        sample.deleteImage();

        sample2.load(TEXT("redimage.bmp"), hWindow_DC);
        sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
        sample2.deleteImage();

        sample3.load(TEXT("blueimage.bmp"), hWindow_DC);
        sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
        sample3.deleteImage();

        DeleteDC(hBmpDC);
        ReleaseDC(hWnd, hWindow_DC);

        //full screen
        SetMenu(hWnd, NULL);
        SetWindowLong(hWnd, GWL_STYLE, WS_VISIBLE | WS_BORDER);
        MoveWindow(
            hWnd,
            0, 0,
            WINDOW_WIDTH, WINDOW_HEIGHT,
            FALSE
        );
        return 0;
    }

    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT:
        // OutputDebugString(L"debug window 2222222\n");
        hWindow_DC = BeginPaint(hWnd, &hPS);
        StretchDIBits(
            hWindow_DC,
            0,
            0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN),
            0,
            0,
            WINDOW_WIDTH * DISPLAY_RESOLUTION,
            WINDOW_HEIGHT * DISPLAY_RESOLUTION,
            lpPixel,
            &hBuffer_bitmapInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );
        EndPaint(hWnd, &hPS);
        return 0;

    case WM_TIMER:
        switch (wParam)
        {
        case REQUEST_ANIMATION_TIMER:
            //fps
            OutputDebugString(L"debug window 2222222\n");
            if (!startFpsCount)
            {
                lastloop = clock();
                startFpsCount = true;
            }
            else
            {
                thisloop = clock();
                fps = 1000 / static_cast<long double>(thisloop - lastloop);
                fps = std::round(fps * 100) / 100;
                lastloop = thisloop;
            }

            PatBlt(
                hBuffer_DC,
                0,
                0,
                WINDOW_WIDTH * DISPLAY_RESOLUTION,
                WINDOW_HEIGHT * DISPLAY_RESOLUTION,
                WHITENESS
            );
            draw(hBuffer_DC, pt_texture_sample);

            InvalidateRect(hWnd, NULL, FALSE);
            return 0;
        case FPS_OUTPUT_TIMER:
            _stprintf_s(mouseMsg, _T("FPS(%4.2lf)[fps]"), fps);
            return 0;
        default:
            OutputDebugStringW(_T("TIMER ERROR\n"));
            return 0;
        }

    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_ESCAPE:
            _stprintf_s(szstr, _T("%s"), _T("ESCAPE"));
            // OutputDebugStringW(_T("ESCAPE\n"));
            break;
        case VK_SPACE:
            _stprintf_s(szstr, _T("%s"), _T("SPACE"));
            //full screen
            SetMenu(hWnd, NULL);
            SetWindowLong(hWnd, GWL_STYLE, WS_VISIBLE | WS_BORDER);
            MoveWindow(
                hWnd,
                0, 0,
                WINDOW_WIDTH, WINDOW_HEIGHT,
                FALSE
            );
            // OutputDebugStringW(_T("SPACE\n"));
            break;
        case VK_SHIFT:
            _stprintf_s(szstr, _T("%s"), _T("SHIFT"));
            // OutputDebugStringW(_T("SHIFT\n"));
            break;
        case 'W':
            _stprintf_s(szstr, _T("%s"), _T("W ON"));
            // OutputDebugStringW(_T("W\n"));
            break;
        default:
            _stprintf_s(szstr, _T("%s"), _T("ANY"));
            break;
        }
        return 0;

    case WM_KEYUP:
        switch (wParam)
        {
        case VK_ESCAPE:
            _stprintf_s(szstr, _T("%s"), _T("NAN"));
            // OutputDebugStringW(_T("ESCAPE UP\n"));
            break;
        case VK_SPACE:
            _stprintf_s(szstr, _T("%s"), _T("NAN"));
            // OutputDebugStringW(_T("SPACE UP\n"));
            break;
        case VK_SHIFT:
            _stprintf_s(szstr, _T("%s"), _T("NAN"));
            // OutputDebugStringW(_T("SHIFT UP\n"));
            break;
        case 'W':
            _stprintf_s(szstr, _T("%s"), _T("W OFF"));
            // OutputDebugStringW(_T("W UP\n"));
            break;
        default:
            _stprintf_s(szstr, _T("%s"), _T("NAN"));
            break;
        }
        return 0;

    case WM_LBUTTONDOWN:
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
        return 0;

    case WM_MOUSEMOVE:
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
        return 0;

    case WM_CLOSE:
        DeleteDC(hBuffer_DC);
        DeleteDC(hBmpDC);

        DeleteObject(hBuffer_bitmap);
        DeleteObject(hBmpFileBitmap);

        DestroyWindow(hWnd);

    case WM_DESTROY:
        hWnd2Open = false;
        PostQuitMessage(0);

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

