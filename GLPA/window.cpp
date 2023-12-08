#include "window.h"

void Window::getFps(){

}

void Window::create(HINSTANCE arghInstance, WINDOW_PROC_TYPE* ptWindowProc){
    wndClass.cbSize = sizeof(wndClass);
    wndClass.style = style;
    wndClass.lpfnWndProc = *ptWindowProc;
    wndClass.cbClsExtra = NULL;
    wndClass.cbWndExtra = NULL;
    wndClass.hInstance = arghInstance;

    wndClass.hIcon = (HICON)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadIcon),
        IMAGE_ICON,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );

    wndClass.hCursor = (HCURSOR)LoadImage(
        NULL, 
        MAKEINTRESOURCE(loadCursor),
        IMAGE_CURSOR,
        0,
        0,
        LR_DEFAULTSIZE | LR_SHARED
    );     
                                                
    wndClass.hbrBackground = (HBRUSH)GetStockObject(backgroundColor);
    wndClass.lpszMenuName = NULL;
    wndClass.lpszClassName = nameApiClass;
    wndClass.hIconSm =
    LoadIcon(wndClass.hInstance, MAKEINTRESOURCE(smallIcon));

    if (!RegisterClassEx(&wndClass)){
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            name,
            MB_ICONEXCLAMATION
        );
    }

    hWnd = CreateWindow(
        nameApiClass,
        name,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        arghInstance,
        NULL
    );

    if (!hWnd){
        MessageBox(
            NULL,
            _T("window make fail"),
            name,
            MB_ICONEXCLAMATION
        );
    }
}

void Window::updateSize(){
    SetWindowPos(hWnd, NULL, 0, 0, 500, 500, SWP_NOMOVE | SWP_NOZORDER);
}

void Window::updateStatus(int argStatus){
    switch (argStatus)
    {
    case WINDOW_STATUS_DEF :
        visible = true;
        ShowWindow(hWnd, SW_SHOWDEFAULT);
        break;

    case WINDOW_STATUS_HIDE :
        visible = false;
        ShowWindow(hWnd, SW_HIDE);
        break;

    case WINDOW_STATUS_MINIMIZE :
        visible = false;
        ShowWindow(hWnd, SW_MINIMIZE);
        break;
    
    default:
        break;
    }
}

bool Window::isVisible(){
    if (visible){
        return true;
    }

    return false;
}

void Window::graphicLoop(){
    if (visible)
    {
        getFps();

        // for(UINT y = 0; y <= height; y++)
        // {
        //     for(UINT x = 0; x <= width; x++)
        //     {
        //         if (x < width && y < height)
        //         {
        //             lpPixel[x+y*width * dpi] = ((DWORD)255 << 24) | ((DWORD)255 << 16) | ((DWORD)0 << 8) | 0;;
        //         }  
        //     }
        // }

        // // 1. 画像の読み込み
        // std::string filename = "image1.png";
        // int imageWidth = 0;
        // int imageHeight = 0;
        // int channels = 0;

        // stbi_uc* pixels = stbi_load(filename.c_str(), &imageWidth, &imageHeight, &channels, STBI_rgb_alpha);

        // if (!pixels) {
        //     // 読み込みエラーが発生した場合の処理
        //     OutputDebugStringW(_T("GLPA : ERROR"));
        // }

        // // 2. ピクセルデータをLPDWORD型変数に変換
        // size_t pixelCount = imageWidth * imageHeight;

        // UINT pixelIndex = 0;
        // UINT imageDrawX = 200;
        // UINT imageDrawY = 300;
        // UINT imageDrawPoint = imageDrawX+ imageDrawY*width * dpi;

        Png temp;

        temp.load("image1.png");

        for(UINT y = 0; y <= imageHeight; y++)
        {
            for(UINT x = 0; x <= imageWidth; x++)
            {
                if (x < imageWidth && y < imageHeight)
                {
                    lpPixel[imageDrawPoint + (x+y*width * dpi)] = (pixels[pixelIndex * 4 + 3] << 24) | 
                                                (pixels[pixelIndex * 4] << 16) | 
                                                (pixels[pixelIndex * 4 + 1] << 8) | 
                                                pixels[pixelIndex * 4 + 2];
                    pixelIndex += 1;
                }  
            }
        }

        // // ピクセルデータの使用が終わったら解放
        // stbi_image_free(pixels);

        InvalidateRect(hWnd, NULL, FALSE);
    }
}

bool Window::minimizeMsg(HWND argHWnd){
    if (argHWnd == hWnd){
        visible = false;
        ShowWindow(hWnd, SW_MINIMIZE);
        return true;
    }
    return false;
}


bool Window::killFocusMsg(HWND argHWnd, bool singleWnd){
    if(argHWnd == hWnd){
        if (singleWnd){
            if(!singleExistence){
                visible = false;
                ShowWindow(hWnd, SW_MINIMIZE);
            }
        }
        else{
            if(minimizeAuto){
                visible = false;
                ShowWindow(hWnd, SW_MINIMIZE);
            }
        }
        
        focus = false;
        return true;
    }
    return false;
}

bool Window::setFocusMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        focus = true;
        visible = true;
        return true;
    }
    
    return false;
}

bool Window::sizeMsg(HWND argHWnd, LPARAM lParam){
    if(argHWnd == hWnd){
        MINMAXINFO* pMinMaxInfo = (MINMAXINFO*)lParam;
        pMinMaxInfo->ptMinTrackSize.x = width;
        pMinMaxInfo->ptMinTrackSize.y = height;
        pMinMaxInfo->ptMaxTrackSize.x = width;
        pMinMaxInfo->ptMaxTrackSize.y = height;
        return true;
    }
    
    return false;
}


bool Window::createMsg(HWND argHWnd){
    if (!created)
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
            (LPVOID*)&lpPixel, 
            NULL, 
            0
        );
        SelectObject(hBufDC, hBufBmp);

        ReleaseDC(hWnd, hWndDC);

        created = true;
        return true;
    }

    return false;
}

bool Window::closeMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        DestroyWindow(hWnd);
        return true;
    }
    
    return false;
}

bool Window::destroyMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        PostQuitMessage(0);
        return true;
    }
    
    return false;
}

bool Window::paintMsg(HWND argHWnd){
    if(argHWnd == hWnd && visible){
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
            lpPixel,
            &hBufBmpInfo,
            DIB_RGB_COLORS,
            SRCCOPY
        );
        EndPaint(hWnd, &hPs);

        return true;
    }
    
    return false;
}

bool Window::userMsg(HWND argHWnd){
    if(argHWnd == hWnd){
        if (focus)
        {

        }
        
        return true;
    }
    
    return false;
}