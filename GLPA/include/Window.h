#ifndef GLPA_WINDOW_H_
#define GLPA_WINDOW_H_

#include <Windows.h>

namespace Glpa {

class Window
{
private :
    HWND hWnd;

public :
    void sendPaintMsg();
};

}

#endif GLPA_WINDOW_H_
