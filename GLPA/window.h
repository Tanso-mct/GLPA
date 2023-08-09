#ifndef WINDOW_H_
#define WINDOW_H_

#include <windows.h>
#include <tchar.h>
#include <time.h>
#include <cmath>
#include <vector>

#include "graphic.h"
#include "userInput.h"

// WINDOW SETTINGS
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)
#define DISPLAY_RESOLUTION 1

// TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2

// FILE TYPE
#define FILETYPE_BMP 0
#define FILETYPE_PNG 1

typedef struct tagWND_STATE
{
    bool foucus = false;
    bool open = false;
} WND_STATE;

typedef struct tagWND_BUFFER
{
    HDC hBufDC;
    HBITMAP hBufBmp;
    BITMAPINFO hBufBmpInfo;
    LPDWORD lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        WINDOW_WIDTH *WINDOW_HEIGHT * 4);
} WND_BUF;

typedef struct tagFPS
{
    int refreshRate;
    bool startFpsCount = false;
    clock_t thisLoopTime;
    clock_t lastLoopTime;
    long double fps;
} WND_FPS;

class WNDMAIN
{
public:
    // Variable for storing arguments of WinMain function
    int nCmdShow;
    HINSTANCE hInstance;

    // Class Registration
    WNDCLASSEX registerClass
    (
        UINT cls_style,
        WNDPROC wnd_proc,
        int cls_extra,
        int wnd_extra,
        HINSTANCE h_instance,
        LPWSTR load_icon,
        LPWSTR load_cursor,
        int background_color,
        LPCWSTR menu_resources_name,
        LPCWSTR cls_name,
        LPWSTR cls_small_icon
    );

    // Check that classes and windows have been created successfully
    int checkClass(WNDCLASSEX *pt_class);
    int checkWindow(HWND created_hWnd);
};

class WINDOW
{
public:
    HWND hWnd;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    int monitorWidth;
    int monitorHeight;
    int displayResolution = 1;

    WND_STATE state;
    WND_BUF buffer;
    WND_FPS fps;
};

class WINDOW_LAU : public WINDOW
{
public:
    int wndWidth = 1200;
    int wndHeight = 800;
    static LRESULT CALLBACK wndProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);
};

class WINDOW_PLAY : public WINDOW
{
public:
    int wndWidth = WINDOW_WIDTH;
    int wndHeight = WINDOW_HEIGHT;
    static LRESULT CALLBACK wndProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);
};

extern WNDMAIN WndMain;
extern WINDOW_LAU WndLAU;
extern WINDOW_PLAY WndPLAY;

#endif WINDOW_H_
