
#include "main.h"
#include "bmp.h"

HWND hWnd;
HWND hWnd2;
HWND gr_hWnd2;
bool hWnd1Open = false;
bool hWnd2Open = false;
int gr_nCmdShow;
HINSTANCE gr_hInstance;

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,           //�A�v���P�[�V�����̃C���X�^���X�n���h��
    _In_opt_ HINSTANCE hPrevInstance,   //�A�v���P�[�V�����ȑO�̃C���X�^���X�n���h��������BWin32�A�v���P�[�V�����ł͏��NULL
    _In_ LPSTR lpCmdLine,               //�R�}���h���C�����i�[���ꂽ�ANULL�ŏI��镶����ւ̃|�C���^������B�v���O�������͊܂܂�Ȃ�
    _In_ int nCmdShow)                  //�E�B���h�E���ǂ̂悤�ɕ\�����邩�̎w�肪����BSW_MESSAGENAME�̒l������  
{
    WNDCLASSEX wcex;        //struct tagWNDCLASSEXW

    wcex.cbSize = sizeof(wcex);                                     //UINT WNDCLASSEX�\���̂̑傫���̐ݒ�
    wcex.style = CS_HREDRAW | CS_VREDRAW;                           //UINT �N���X�X�^�C����\���BCS_MESSAGENAME�̒l��OR���Z�q�őg�ݍ��킹���l�ƂȂ�
    wcex.lpfnWndProc = WndProc;                                     //WNDPROC WNDPROC���w���|�C���^
    wcex.cbClsExtra = 0;                                            //int �E�B���h�E�N���X�\���̂̐ՂɊ��蓖�Ă�o�C�g��������
    wcex.cbWndExtra = 0;                                            //int �E�B���h�E�C���X�^���X�̐ՂɊ��蓖�Ă�o�C�g��������
    wcex.hInstance = hInstance;                                     //HINSTANCE �C���X�^���X�n���h��
    wcex.hIcon =                                                    //HICON �N���X�A�C�R�����w�肷��
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex.hCursor =                                                  //HCURSOR �N���X�J�[�\�����w�肷��
        LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH �N���X�w�i�u���V���w�肷��
    wcex.lpszMenuName = NULL;                                       //LPCSTR �N���X���j���[�̃��\�[�X�����w�肷��
    wcex.lpszClassName = L"window1";                                //LPCSTR �E�B���h�E�N���X�̖��O���w�肷��
    wcex.hIconSm =                                                  //HICON �����ȃN���X�A�C�R�����w�肷��
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

    wcex2.cbSize = sizeof(wcex2);                                    //UINT WNDCLASSEX�\���̂̑傫���̐ݒ�
    wcex2.style = CS_HREDRAW | CS_VREDRAW;                           //UINT �N���X�X�^�C����\���BCS_MESSAGENAME�̒l��OR���Z�q�őg�ݍ��킹���l�ƂȂ�
    wcex2.lpfnWndProc = WndProc2;                                    //WNDPROC WNDPROC���w���|�C���^
    wcex2.cbClsExtra = 0;                                            //int �E�B���h�E�N���X�\���̂̐ՂɊ��蓖�Ă�o�C�g��������
    wcex2.cbWndExtra = 0;                                            //int �E�B���h�E�C���X�^���X�̐ՂɊ��蓖�Ă�o�C�g��������
    wcex2.hInstance = hInstance;                                     //HINSTANCE �C���X�^���X�n���h��
    wcex2.hIcon =                                                    //HICON �N���X�A�C�R�����w�肷��
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex2.hCursor =                                                  //HCURSOR �N���X�J�[�\�����w�肷��
        LoadCursor(NULL, IDC_ARROW);
    wcex2.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH �N���X�w�i�u���V���w�肷��
    wcex2.lpszMenuName = NULL;                                       //LPCSTR �N���X���j���[�̃��\�[�X�����w�肷��
    wcex2.lpszClassName = L"window2";                                //LPCSTR �E�B���h�E�N���X�̖��O���w�肷��
    wcex2.hIconSm =                                                  //HICON �����ȃN���X�A�C�R�����w�肷��
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

    HWND hWnd = CreateWindow(           //HWND �E�B���h�E�n���h��
        L"window1",                     //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
        L"GLPA",                        //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
        WS_OVERLAPPEDWINDOW,            //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
        CW_USEDEFAULT, CW_USEDEFAULT,   //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
        WINDOW_WIDTH, WINDOW_HEIGHT,    //int �E�B���h�E�̕�, �E�B���h�E�̍���
        HWND_DESKTOP,                   //HWND �e�E�B���h�E�̃n���h��
        NULL,                           //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
        hInstance,                      //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
        NULL                            //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
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

    MSG msg;        //���b�Z�[�W�\����

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return (int)msg.wParam;             //�֐���WM_QUIT���b�Z�[�W���󂯎���ďI�������Ƃ��́A���b�Z�[�W��wParam�p�����[�^�����I���R�[�h��Ԃ��B
							            //�֐������b�Z�[�W���[�v�ɓ���O�ɏI�������Ƃ��́A�O��Ԃ�
}

void draw(HDC hMemDC, TEXTURE *texture)
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
	SelectObject(hMemDC, hFont1);  
    
    TextOut(
        hMemDC,
        pt.x,
        pt.y,
        mouseMsg,
        _tcslen(mouseMsg)
    );
    DeleteObject(hFont1); 

    TextOut(
        hMemDC,
        5,
        5,
        szstr,
        _tcslen(szstr)
    );
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    //win32 define
    PAINTSTRUCT ps;
    HDC hdc;

    //bufer bmp dc
    static HBITMAP hBitmap;    
    static HDC hMemDC;
    static BITMAPINFO bmpInfo; 

    // //bmpfile dc
    // static HBITMAP hBmpFileBitmap;    
    // static HDC hBmpDC;
    // static BITMAPINFO bmpFileInfo; 

    //textute
    static TEXTURE texture_sample;
    static TEXTURE *pt_texture_sample = &texture_sample;

    //bmpfile
    static BMPFILE sample;    
    static BMPFILE *pt_sample = &sample;

    static BMPFILE sample2;    
    static BMPFILE *pt_sample2 = &sample2;

    static BMPFILE sample3;    
    static BMPFILE *pt_sample3 = &sample3;

    //
    
    //fps
    static int refreshRate;
    static bool startFpsCount = false;
    static clock_t thisloop;
    static clock_t lastloop;
    static long double fps;

    if (hWnd2Open && message != WM_SETFOCUS)
    {
        return DefWindowProc(hWnd, message, wParam, lParam);;
    }
    else if (message == WM_SETFOCUS)
    {
        hWnd1Open = true;
        hWnd2Open = false;
    }

    switch (message)
    {
        case WM_CREATE :
            {
                hdc = GetDC(hWnd);
                refreshRate = GetDeviceCaps(hdc, VREFRESH);
                ReleaseDC(hWnd, hdc);

                SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / refreshRate), NULL);
                SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);
                
                //bmp buffer dc
                bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                bmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                bmpInfo.bmiHeader.biPlanes = 1;
                bmpInfo.bmiHeader.biBitCount = 32;
                bmpInfo.bmiHeader.biCompression = BI_RGB;
                
                hdc = GetDC(hWnd);
                hMemDC = CreateCompatibleDC(hdc);
                hBitmap = CreateDIBSection(NULL, &bmpInfo, DIB_RGB_COLORS, (LPVOID*)&lpPixel, NULL, 0);
                SelectObject(hMemDC, hBitmap);
                SelectObject(hMemDC, GetStockObject(DC_PEN));
                SelectObject(hMemDC, GetStockObject(DC_BRUSH)); 
                
                //bmp file dc
                // bmpFileInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                // bmpFileInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
                // bmpFileInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;      
                // bmpFileInfo.bmiHeader.biPlanes = 1;
                // bmpFileInfo.bmiHeader.biBitCount = 32;
                // bmpFileInfo.bmiHeader.biCompression = BI_RGB;
                
                // hBmpDC = CreateCompatibleDC(hdc);
                // hBmpFileBitmap = CreateDIBSection(NULL, &bmpFileInfo, DIB_RGB_COLORS, (LPVOID*)&bmpPixel, NULL, 0);
                // SelectObject(hBmpDC, hBmpFileBitmap);
                
                //load texture
                sample.load(TEXT("sample.bmp"), hdc);
                sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hMemDC, lpPixel);
                texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
                sample.deleteImage(); 

                sample2.load(TEXT("redimage.bmp"), hdc);
                sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hMemDC, lpPixel);
                texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
                sample2.deleteImage();   

                sample3.load(TEXT("blueimage.bmp"), hdc);
                sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hMemDC, lpPixel);
                texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
                sample3.deleteImage();     

                // DeleteDC(hBmpDC);
                ReleaseDC(hWnd, hdc);

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
                // OutputDebugString(L"debug window 1111111\n");
                hdc = BeginPaint(hWnd, &ps);
                StretchDIBits(
                    hdc,
                    0,
                    0,
                    GetSystemMetrics(SM_CXSCREEN),
                    GetSystemMetrics(SM_CYSCREEN), 
                    0,
                    0,
                    WINDOW_WIDTH * DISPLAY_RESOLUTION,
                    WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                    lpPixel,
                    &bmpInfo,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
                EndPaint(hWnd, &ps);
                return 0;
                
        case WM_TIMER :
            switch (wParam)
            {
                case REQUEST_ANIMATION_TIMER :
                        //fps
                        OutputDebugString(L"debug window 1111111\n");
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
                            hMemDC, 
                            0, 
                            0, 
                            WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                            WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                            WHITENESS
                        );
                        draw(hMemDC, pt_texture_sample);

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
                            hWnd2 = CreateWindow(           //HWND �E�B���h�E�n���h��
                                L"window2",                     //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
                                L"GLPA2",                        //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
                                WS_OVERLAPPEDWINDOW,            //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
                                CW_USEDEFAULT, CW_USEDEFAULT,   //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
                                WINDOW_WIDTH, WINDOW_HEIGHT,    //int �E�B���h�E�̕�, �E�B���h�E�̍���
                                HWND_DESKTOP,                           //HWND �e�E�B���h�E�̃n���h��
                                NULL,                           //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
                                gr_hInstance,                      //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
                                NULL                            //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
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
                pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
                pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
                // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
                return 0;

        case WM_MOUSEMOVE :
                pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
                pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
                // _stprintf_s(mouseMsg, _T("%d,%d"), pt.x, pt.y);
                return 0;

        case WM_CLOSE :
                DeleteDC(hMemDC);
                // DeleteDC(hBmpDC);

                DeleteObject(hBitmap);
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
    PAINTSTRUCT ps;
    HDC hdc;

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
    static HBITMAP hBitmap;
    static HDC hMemDC;
    static BITMAPINFO bmpInfo;

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
        hdc = GetDC(hWnd);
        refreshRate = GetDeviceCaps(hdc, VREFRESH);
        ReleaseDC(hWnd, hdc);

        SetTimer(hWnd, REQUEST_ANIMATION_TIMER, std::floor(1000 / refreshRate), NULL);
        SetTimer(hWnd, FPS_OUTPUT_TIMER, 250, NULL);

        //bmp buffer dc
        bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmpInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
        bmpInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;
        bmpInfo.bmiHeader.biPlanes = 1;
        bmpInfo.bmiHeader.biBitCount = 32;
        bmpInfo.bmiHeader.biCompression = BI_RGB;

        hdc = GetDC(hWnd);
        hMemDC = CreateCompatibleDC(hdc);
        hBitmap = CreateDIBSection(NULL, &bmpInfo, DIB_RGB_COLORS, (LPVOID*)&lpPixel, NULL, 0);
        SelectObject(hMemDC, hBitmap);
        SelectObject(hMemDC, GetStockObject(DC_PEN));
        SelectObject(hMemDC, GetStockObject(DC_BRUSH));

        //bmp file dc
        bmpFileInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmpFileInfo.bmiHeader.biWidth = +WINDOW_WIDTH * DISPLAY_RESOLUTION;
        bmpFileInfo.bmiHeader.biHeight = -WINDOW_HEIGHT * DISPLAY_RESOLUTION;
        bmpFileInfo.bmiHeader.biPlanes = 1;
        bmpFileInfo.bmiHeader.biBitCount = 32;
        bmpFileInfo.bmiHeader.biCompression = BI_RGB;

        hBmpDC = CreateCompatibleDC(hdc);
        hBmpFileBitmap = CreateDIBSection(NULL, &bmpFileInfo, DIB_RGB_COLORS, (LPVOID*)&bmpPixel, NULL, 0);
        SelectObject(hBmpDC, hBmpFileBitmap);

        //load texture
        sample.load(TEXT("sample.bmp"), hdc);
        sample.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample.pixel, sample.getWidth(), sample.getHeight());
        sample.deleteImage();

        sample2.load(TEXT("redimage.bmp"), hdc);
        sample2.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample2.pixel, sample2.getWidth(), sample2.getHeight());
        sample2.deleteImage();

        sample3.load(TEXT("blueimage.bmp"), hdc);
        sample3.create(WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, hBmpDC, bmpPixel);
        texture_sample.insertBMP(sample3.pixel, sample3.getWidth(), sample3.getHeight());
        sample3.deleteImage();

        DeleteDC(hBmpDC);
        ReleaseDC(hWnd, hdc);

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
        hdc = BeginPaint(hWnd, &ps);
        StretchDIBits(
            hdc,
            0,
            0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN),
            0,
            0,
            WINDOW_WIDTH * DISPLAY_RESOLUTION,
            WINDOW_HEIGHT * DISPLAY_RESOLUTION,
            lpPixel,
            &bmpInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );
        EndPaint(hWnd, &ps);
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
                hMemDC,
                0,
                0,
                WINDOW_WIDTH * DISPLAY_RESOLUTION,
                WINDOW_HEIGHT * DISPLAY_RESOLUTION,
                WHITENESS
            );
            draw(hMemDC, pt_texture_sample);

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
        DeleteDC(hMemDC);
        DeleteDC(hBmpDC);

        DeleteObject(hBitmap);
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

