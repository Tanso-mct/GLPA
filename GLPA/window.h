#ifndef WINDOW_H_
#define WINDOW_H_

#include <windows.h>
#include <tchar.h>
#include <time.h>

//WINDOW SETTINGS
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)
#define DISPLAY_RESOLUTION 1

//TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2


typedef struct GR_WINDOW_VARIABLE
{
    // window
    HWND hWnd;
    bool foucus = false;
    bool open = false;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    // buffer
    HDC hBufDC;
    HBITMAP hBufBmp;    
    BITMAPINFO hBufBmpInfo;      
    LPDWORD lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        WINDOW_WIDTH*WINDOW_HEIGHT*4
    );

    // fps
    int refreshRate;
    bool startFpsCount = false;
    clock_t thisLoopTime;
    clock_t lastLoopTime;
    long double fps;
} GR_WNDVARI;

// Grobal window structre
extern GR_WNDVARI WND_LAU;
extern GR_WNDVARI WND_PLAY;

// Stores Winmain function arguments as global variables
extern int gr_nCmdShow;
extern HINSTANCE gr_hInstance;

// Window Procedure
LRESULT CALLBACK WndProc_LAU(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK WndProc_PLAY(HWND, UINT, WPARAM, LPARAM);

#endif WINDOW_H_
