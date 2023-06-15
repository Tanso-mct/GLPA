#ifndef MAIN_H_
#define MAIN_H_

#define STRICT
#include <windows.h>
#include <tchar.h>
#include <cmath>
#include <time.h>

// HWND hWnd;
HWND hWnd_PLAY;
HWND gr_hWnd2;
bool hWnd1Open = false;
bool hWnd2Open = false;
int gr_nCmdShow;
bool hWnd_LAU_foucus = true;
HINSTANCE gr_hInstance;

//WINDOW SETTINGS
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)
#define DISPLAY_RESOLUTION 1

//double buffer pixel
LPDWORD lpPixel = 
(LPDWORD)HeapAlloc
(
    GetProcessHeap(),
    HEAP_ZERO_MEMORY,
    WINDOW_WIDTH*WINDOW_HEIGHT*4
);

//bmpfile load pixel
LPDWORD bmpPixel = 
(LPDWORD)HeapAlloc
(
    GetProcessHeap(),
    HEAP_ZERO_MEMORY,
    WINDOW_WIDTH*WINDOW_HEIGHT*4
);

//Window Procedure
LRESULT CALLBACK WndProc_LAU(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK WndProc2(HWND, UINT, WPARAM, LPARAM);

//TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2

TCHAR szstr[256] = _T("NAN KEY");
TCHAR mouseMsg[256] = _T("NOW COORDINATE");

POINT pt = {
    5,
    20
};

#endif // MAIN_H_
