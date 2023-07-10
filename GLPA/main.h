#ifndef MAIN_H_
#define MAIN_H_

#include <windows.h>
#include <tchar.h>
#include <cmath>
#include <time.h>

//WINDOW SETTINGS
#define WINDOW_WIDTH GetSystemMetrics(SM_CXSCREEN)
#define WINDOW_HEIGHT GetSystemMetrics(SM_CYSCREEN)
#define DISPLAY_RESOLUTION 1

//TIMER
#define REQUEST_ANIMATION_TIMER 1
#define FPS_OUTPUT_TIMER 2

//Stores Winmain function arguments as global variables
extern int gr_nCmdShow;
extern HINSTANCE gr_hInstance;

//Window Procedure
LRESULT CALLBACK WndProc_LAU(HWND, UINT, WPARAM, LPARAM);
// LRESULT CALLBACK WndProc2(HWND, UINT, WPARAM, LPARAM);

extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];

extern POINT pt;

#include "window.h"
#include "bmp.h"
// #include "graphic.h"
#include "userinput.h"


#endif // MAIN_H_
