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

//Stores Winmain function arguments as global variables
extern int gr_nCmdShow;
extern HINSTANCE gr_hInstance;

//Window Procedure
LRESULT CALLBACK WndProc_LAU(HWND, UINT, WPARAM, LPARAM);
// LRESULT CALLBACK WndProc2(HWND, UINT, WPARAM, LPARAM);

extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];

extern POINT pt;

#endif WINDOW_H_
