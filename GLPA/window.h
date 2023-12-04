#ifndef WINDOW_H_
#define WINDOW_H_

#include <Windows.h>
#include <tchar.h>
#include <string>
#include <unordered_map>

#include "fps.h"

#define DEF_WINDOW_NAME L"Default"
#define DEF_WINDOW_CLASS_NAME L"Default"
#define DEF_WINDOW_WIDTH 1200
#define DEF_WINDOW_HEIGHT 800
#define DEF_WINDOW_DPI  1.0
#define DEF_WINDOW_FULLSCREEN_TOGGLE false
#define DEF_WINDOW_STYLE CS_HREDRAW | CS_VREDRAW
#define DEF_WINDOW_LOAD_ICON IDI_APPLICATION
#define DEF_WINDOW_LOAD_CURSOR IDC_ARROW
#define DEF_WINDOW_BACKGROUND_COLOR WHITE_BRUSH
#define DEF_WINDOW_SMALL_ICON IDI_APPLICATION

using WINDOW_PROC_TYPE = LRESULT CALLBACK(HWND, UINT, WPARAM, LPARAM);

class Window
{
public :
    Window(
        LPCWSTR argName = DEF_WINDOW_NAME, 
        LPCWSTR argNameApiClass = DEF_WINDOW_CLASS_NAME, 
        double argWidth = DEF_WINDOW_WIDTH, 
        double argHeight = DEF_WINDOW_HEIGHT, 
        double argDpi = DEF_WINDOW_DPI,
        double argMaxFps = DEF_MAX_FPS,
        UINT argWndStyle = DEF_WINDOW_STYLE,
        LPWSTR argLoadIcon = DEF_WINDOW_LOAD_ICON, 
        LPWSTR argLoadCursor = DEF_WINDOW_LOAD_CURSOR,
        int argBackgroundColor = DEF_WINDOW_BACKGROUND_COLOR,
        LPWSTR argSmallIcon = DEF_WINDOW_SMALL_ICON
    ) : status(0),
        focus(false),
        name(argName), 
        nameApiClass(argNameApiClass), 
        width(argWidth), 
        height(argHeight), 
        dpi(argDpi), 
        style(argWndStyle), 
        loadIcon(argLoadIcon), 
        loadCursor(argLoadCursor),
        backgroundColor(argBackgroundColor), 
        smallIcon(argSmallIcon),
        hWndDC(nullptr),
        hBufDC(nullptr),
        hBufBmp(nullptr),
        lpPixel(nullptr) {
        fps.max = argMaxFps;
        lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        width *height * 4);
    }

    void create(HINSTANCE arg_h_instance, WINDOW_PROC_TYPE* pt_window_proc);
    void show();
    void hide();

    bool killFoucusMsg();
    bool setFoucusMsg();
    bool createMsg();
    bool closeMsg();
    bool destroyMsg();
    bool paintMsg();
    bool userMsg();

    // double getFps();
    void changeSize();
    // void copyArgBuffer();

    int status = 0;
    bool focus = false;
    bool fullScreenToggle;

    LPCWSTR name;
    LPCWSTR nameApiClass;

    double width;
    double height;
    double dpi;

    UINT style;
    LPWSTR loadIcon;
    LPWSTR loadCursor;
    int backgroundColor;
    LPWSTR smallIcon;

    HWND hWnd;
    HWND createdHWND = nullptr;

    WNDCLASSEX wndClass;

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