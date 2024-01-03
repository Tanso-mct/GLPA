#include "window.h"

void Window::getFps(){

}

void Window::create(HINSTANCE arghInstance, GLPA_WINDOW_PROC_TYPE* ptWindowProc, DWORD viewStyle){
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
        throw std::runtime_error(ERROR_WINDOW_REGISTER_CLASS);
    }


    hWnd = CreateWindow(
        nameApiClass,
        name,
        viewStyle,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        NULL,
        NULL,
        arghInstance,
        NULL
    );

    if (!hWnd){
        throw std::runtime_error(ERROR_WINDOW_CREATE);
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

        if (ptScene->names[useScene] == GLPA_SCENE_2D){
            ptScene->data2d[useScene].edit(hBufDC, lpPixel, width, height, dpi);
            ptScene->data2d[useScene].update(hBufDC, lpPixel, width, height, dpi);
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


void Window::setScene(Scene *argPtScene, std::string scName){
    ptScene = argPtScene;
    useScene = scName;

    if(ptScene->data3d.find(scName) != ptScene->data3d.end()){
        
    }
    else if (ptScene->data2d.find(scName) != ptScene->data2d.end()){
        ptScene->data2d[scName].storeUseWndParam(width, height, dpi);
    }

}
