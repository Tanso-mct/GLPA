#ifndef WINDOW_H_
#define WINDOW_H_

#include <windows.h>
#include <string>

#include "fps.h"

class Window
{
public :
    double getFps();
    void changeSize();
    static LRESULT CALLBACK procedure(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

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

};

#endif WINDOW_H_