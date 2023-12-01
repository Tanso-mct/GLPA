#ifndef WINDOW_H_
#define WINDOW_H_

#include <Windows.h>
#include <string>

#include "fps.h"

#define DEF_WINDOW_NAME "Default"
#define DEF_WINDOW_CLASS_NAME "Default"
#define DEF_WINDOW_WIDTH 1200
#define DEF_WINDOW_HEIGHT 800
#define DEF_WINDOW_DPI  1.0
#define DEF_WINDOW_FULLSCREEN_TOGGLE false
#define DEF_WINDOW_STYLE CS_HREDRAW | CS_VREDRAW
#define DEF_WINDOW_LOAD_ICON IDI_APPLICATION
#define DEF_WINDOW_LOAD_CURSOR IDC_ARROW
#define DEF_WINDOW_BACKGROUND_COLOR WHITE_BRUSH
#define DEF_WINDOW_SMALL_ICON IDI_APPLICATION

class Window
{
public :
    Window
    (
        std::string argName = DEF_WINDOW_NAME, 
        std::string argNameApiClass = DEF_WINDOW_CLASS_NAME, 
        double argWidth = DEF_WINDOW_WIDTH, 
        double argHeight = DEF_WINDOW_HEIGHT, 
        double argDpi = DEF_WINDOW_DPI,
        double argMaxFps = DEF_MAX_FPS,
        bool argFullScreenToggle = DEF_WINDOW_FULLSCREEN_TOGGLE,
        UINT argWndStyle = DEF_WINDOW_STYLE,
        LPWSTR argLoadIcon = DEF_WINDOW_LOAD_ICON, 
        LPWSTR argLoadCursor = DEF_WINDOW_LOAD_CURSOR,
        int argBackgroundColor = DEF_WINDOW_BACKGROUND_COLOR,
        LPWSTR argSmallIcon = DEF_WINDOW_SMALL_ICON
    ) : existence(false),
        foucus(false),
        name(argName), 
        nameApiClass(argNameApiClass), 
        width(argWidth), 
        height(argHeight), 
        dpi(argDpi), 
        fullScreenToggle(argFullScreenToggle), 
        style(argWndStyle), 
        loadIcon(argLoadIcon), 
        loadCursor(argLoadCursor),
        backgroundColor(argBackgroundColor), 
        smallIcon(argSmallIcon),
        hWndDC(nullptr),
        hBufDC(nullptr),
        hBufBmp(nullptr),
        lpPixel(nullptr)
{
        fps.max = argMaxFps;
        lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        width *height * 4);
}

    // double getFps();
    // void changeSize();
    // void copyArgBuffer();
    static LRESULT CALLBACK procedure(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

    bool existence = false;
    bool foucus = false;

    std::string name;
    std::string nameApiClass;

    double width;
    double height;
    double dpi;

    UINT style;
    LPWSTR loadIcon;
    LPWSTR loadCursor;
    int backgroundColor;
    LPWSTR smallIcon;
    bool fullScreenToggle;

    HWND hWnd;

private :
    Fps fps;

    HDC hWndDC;
    PAINTSTRUCT hPs;

    HDC hBufDC;
    HBITMAP hBufBmp;
    BITMAPINFO hBufBmpInfo;
    LPDWORD lpPixel;


};

#endif WINDOW_H_