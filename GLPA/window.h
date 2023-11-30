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
        bool argFullScreenToggle
    )
    {
        name = argName;
        nameApiClass = argNameApiClass;
        width = argWidth;
        height = argHeight;
        dpi = argDpi;
        fps.max = argMaxFps;
        fullScreenToggle = argFullScreenToggle;

        lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        width *height * 4);
    }

    double getFps();
    void changeSize();
    void copyArgBuffer();
    static LRESULT CALLBACK procedure(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

private :
    bool existence = false;
    bool foucus = false;

    std::string name;
    std::string nameApiClass;

    double width;
    double height;
    double dpi;

    Fps fps;

    bool fullScreenToggle;

    HWND hWnd;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    HDC hBufDC;
    HBITMAP hBufBmp;
    BITMAPINFO hBufBmpInfo;
    LPDWORD lpPixel;


};

#endif WINDOW_H_