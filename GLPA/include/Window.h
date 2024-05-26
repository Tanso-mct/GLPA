#ifndef GLPA_WINDOW_H_
#define GLPA_WINDOW_H_

#include <Windows.h>

namespace Glpa {

class Window
{
private :
    LPCWSTR name;
    LPCWSTR apiClassName;

    HWND hWnd;

    int width;
    int height;
    int dpi;

    UINT style;
    LPWSTR loadIcon;
    LPWSTR loadCursor;
    int bgColor;
    LPWSTR smallIcon;

    WNDCLASSEX apiClass;

    HDC hWndDC = nullptr;
    PAINTSTRUCT hPs;

    HDC hBufDC = nullptr;
    HBITMAP hBufBmp = nullptr;
    BITMAPINFO hBufBmpInfo;

    LPDWORD pixels;

public :
    Window
    (
        LPCWSTR argName, LPCWSTR argApiClassName, int argWidth, int argHeight, int argDpi,
        UINT argStyle, LPWSTR argLoadIcon, LPWSTR argLoadCursor, int argBackgroundColor, LPWSTR argSmallIcon
    );



public :
    void sendPaintMsg();
};

}

#endif GLPA_WINDOW_H_
