#include "window.h"

void Window::getFps(){

}

void Window::create(HINSTANCE arghInstance, GLPA_WINDOW_PROC_TYPE* ptWindowProc){
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
    case GLPA_WINDOW_STATUS_DEF :
        visible = true;
        ShowWindow(hWnd, SW_SHOWDEFAULT);
        break;

    case GLPA_WINDOW_STATUS_HIDE :
        visible = false;
        ShowWindow(hWnd, SW_HIDE);
        break;

    case GLPA_WINDOW_STATUS_MINIMIZE :
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

        std::size_t imageDrawX = 200;
        std::size_t imageDrawY = 300;
        std::size_t imageDrawPoint = imageDrawX+ imageDrawY*width * dpi;

        Png temp;
        temp.load("resource/scene/Launcher_load/image1.png");

        for(std::size_t y = 0; y <= temp.height; y++)
        {
            for(std::size_t x = 0; x <= temp.width; x++)
            {
                if (x < temp.width && y < temp.height)
                {
                    lpPixel[imageDrawPoint + (x+y*width * dpi)] = temp.data[x+y*temp.width];
                }  
            }
        }
        
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