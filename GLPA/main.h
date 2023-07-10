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

#include "window.h"
#include "bmp.h"
// #include "graphic.h"
#include "userinput.h"


#endif // MAIN_H_
