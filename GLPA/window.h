#ifndef WINDOW_H_
#define WINDOW_H_

#include <Windows.h>
#include <string>

#include "fps.h"

class Window
{
public :
    Window
    (
        std::string argName, 
        std::string argNameApiClass, 
        double argWidth, 
        double argHeight, 
        double argDpi,
        double argMaxFps,
        bool argFullScreenToggle,
        UINT argWndStyle,
        LPWSTR argLoadIcon, 
        LPWSTR argLoadCursor,
        int argBackgroundColor,
        LPWSTR argSmallIcon
    )
    {
        name = argName;
        nameApiClass = argNameApiClass;
        width = argWidth;
        height = argHeight;
        dpi = argDpi;
        fps.max = argMaxFps;
        fullScreenToggle = argFullScreenToggle;

        style = argWndStyle;
        loadIcon = argLoadIcon;
        loadCursor = argLoadCursor;
        backgroundColor = argBackgroundColor;
        smallIcon = argSmallIcon;


        lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        width *height * 4);
    }

    // double getFps();
    // void changeSize();
    // void copyArgBuffer();
    static LRESULT CALLBACK procedure(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

    bool existence = false;
    bool foucus = false;

    std::string name;
    std::string nameApiClass;

    double width;
    double height;
    double dpi;

    UINT style;
    LPWSTR loadIcon;
    LPWSTR loadCursor;
    int backgroundColor;
    LPWSTR smallIcon;
    bool fullScreenToggle;

    HWND hWnd;

private :
    Fps fps;

    HDC hWndDC;
    PAINTSTRUCT hPs;

    HDC hBufDC;
    HBITMAP hBufBmp;
    BITMAPINFO hBufBmpInfo;
    LPDWORD lpPixel;


};

#endif WINDOW_H_