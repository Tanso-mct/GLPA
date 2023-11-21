#ifndef WINDOW_H_
#define WINDOW_H_

#include <windows.h>
#include <string>

#include "fps.h"
#include "launcher.h"

class Window
{
public :
    double getFps();
    void changeSize();

private :
    bool existence = false;
    bool foucus = false;

    double width;
    double heiht;
    double dpi;

    Fps fps;

    HWND hWnd;
    HDC hWndDC;
    PAINTSTRUCT hPs;

    Launcher launcherWnd;
};

#endif WINDOW_H_