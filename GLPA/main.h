#ifndef MAIN_H_
#define MAIN_H_

#include <windows.h>
#include <tchar.h>
#include <cmath>
#include <time.h>

//Stores Winmain function arguments as global variables
int gr_nCmdShow;
HINSTANCE gr_hInstance;

//WINDOW SETTINGS
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)
#define DISPLAY_RESOLUTION 1

//TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2

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

GR_WNDVARI WND_LAU;
// GR_WNDVARI WND_PLAY;

// HWND hWnd_LAU;
// HWND hWnd_PLAY;
// HWND hWnd2;
// HWND gr_hWnd2;
// bool hWnd1Open = false;
// bool hWnd2Open = false;
// bool hWnd_LAU_foucus = true;

//double buffer pixel
// LPDWORD lpPixel = 
// (LPDWORD)HeapAlloc
// (
//     GetProcessHeap(),
//     HEAP_ZERO_MEMORY,
//     WINDOW_WIDTH*WINDOW_HEIGHT*4
// );

//bmpfile load pixel
// LPDWORD bmpPixel = 
// (LPDWORD)HeapAlloc
// (
//     GetProcessHeap(),
//     HEAP_ZERO_MEMORY,
//     WINDOW_WIDTH*WINDOW_HEIGHT*4
// );

//Window Procedure
LRESULT CALLBACK WndProc_LAU(HWND, UINT, WPARAM, LPARAM);
// LRESULT CALLBACK WndProc2(HWND, UINT, WPARAM, LPARAM);

TCHAR szstr[256] = _T("NAN KEY");
TCHAR mouseMsg[256] = _T("NOW COORDINATE");

POINT pt = {
    5,
    20
};

#endif // MAIN_H_
