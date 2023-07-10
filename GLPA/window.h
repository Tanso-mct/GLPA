#ifndef WINDOW_H_
#define WINDOW_H_

#include "main.h"

#include <windows.h>
#include <tchar.h>
#include <time.h>

typedef struct GR_WINDOW_VARIABLE
{
    //window
    HWND hWnd;
    bool foucus = false;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    //buffer
    HDC hBufDC;
    HBITMAP hBufBmp;    
    BITMAPINFO hBufBmpInfo;      
    LPDWORD lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        WINDOW_WIDTH*WINDOW_HEIGHT*4
    );

    //fps
    int refreshRate;
    bool startFpsCount = false;
    clock_t thisLoopTime;
    clock_t lastLoopTime;
    long double fps;
} GR_WNDVARI;

extern GR_WNDVARI WND_LAU;

#endif WINDOW_H_
