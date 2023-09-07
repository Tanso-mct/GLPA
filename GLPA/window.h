#ifndef WINDOW_H_
#define WINDOW_H_

#include <windows.h>
#include <tchar.h>
#include <time.h>
#include <thread>
#include <chrono>
#include <cmath>

#include "graphic.h"
#include "file.h"
#include "userInput.h"

// WINDOW SETTINGS
#define DISPLAY_RESOLUTION 1
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)

// TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2

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

// typedef struct tagFPS
// {
//     int refreshRate;
//     bool startFpsCount = false;
//     clock_t thisLoopTime;
//     clock_t lastLoopTime;
//     double currentFps;

// } WND_FPS;

class FPS_CALC
{
public :
    double fps;
    double setFps = 1000;
    double currentFps;
    double maxFps;
    clock_t thisLoopTime;
    clock_t lastLoopTime;
    double nextLoopTime;
    bool startFpsCalc = false;

    void fpsLimiter();
};

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
    bool checkClass(WNDCLASSEX *pt_class);
    bool checkWindow(HWND created_hWnd);
};

class WINDOW
{
public:
    HWND hWnd;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    // SIZE2 monitor;
    int displayRes = 1;

    WND_STATE state;
    WND_BUF buffer;
    FPS_CALC fpsSystem;
};

class WINDOW_LAU : public WINDOW
{
public:
    SIZE2 windowSize = {1200, 800};
    static LRESULT CALLBACK wndProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);
};

class WINDOW_PLAY : public WINDOW
{
public:
    SIZE2 windowSize = {WINDOW_WIDTH, WINDOW_HEIGHT};
    static LRESULT CALLBACK wndProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);
};

extern WNDMAIN WndMain;
extern WINDOW_LAU WndLAU;
extern WINDOW_PLAY WndPLAY;

#endif WINDOW_H_
