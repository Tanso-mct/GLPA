#include "glpa.h"

void Glpa::createWindow
(
    std::string wndName,
    std::string wndApiClassName,
    double wndWidth,
    double wndHeight,
    double wndDpi,
    double wndMaxFps,
    bool wndFullScreen
)
{
    Window newWnd(wndName, wndApiClassName, wndWidth, wndHeight);
    window.emplace(wndName, newWnd);
}
