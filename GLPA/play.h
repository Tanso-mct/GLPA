#ifndef PLAY_H_
#define PLAY_H_

#include "window.h"
#include "buffer_3d.h"

class Play
{
public :
    void update();

    static LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM w_param, LPARAM l_param);

private :
    Buffer3d buffer;

};

#endif  PLAY_H_


