#ifndef LAUNCHER_H_
#define LAUNCHER_H_

#include "window.h"
#include "window_api.h"
#include "play.h"
#include "buffer_2d.h"

class Launcher
{
public :
    void update();

    bool loadingDataLoad();
    bool useDataLoad();

    static LRESULT CALLBACK wndProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

private :
    WindowApi winApi;
    Play playWnd;
    Buffer2d buffer;
};

#endif LAUNCHER_H_
