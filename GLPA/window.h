/**
 * @file window.h
 * @brief Describes window-related processing. Set up a scene and draw it.
 * @author Tanso-mct
 * @date 2023-12-7
 */

#ifndef WINDOW_H_
#define WINDOW_H_

#include <Windows.h>
#include <tchar.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "fps.h"

#define DEF_WINDOW_NAME L"Default"
#define DEF_WINDOW_CLASS_NAME L"Default"
#define DEF_WINDOW_WIDTH 1200
#define DEF_WINDOW_HEIGHT 800
#define DEF_WINDOW_DPI  1.0
#define DEF_WINDOW_STYLE CS_HREDRAW | CS_VREDRAW
#define DEF_WINDOW_LOAD_ICON IDI_APPLICATION
#define DEF_WINDOW_LOAD_CURSOR IDC_ARROW
#define DEF_WINDOW_BACKGROUND_COLOR WHITE_BRUSH
#define DEF_WINDOW_SMALL_ICON IDI_APPLICATION

using WINDOW_PROC_TYPE = LRESULT CALLBACK(HWND, UINT, WPARAM, LPARAM);

#define WINDOW_STATUS_DEF 0
#define WINDOW_STATUS_HIDE 1
#define WINDOW_STATUS_BORDERLESS_SCREEN 2
#define WINDOW_STATUS_FULL_SCREEN 3
#define WINDOW_STATUS_MINIMIZE 4

class Window
{
public :
    Window(
        LPCWSTR argName = DEF_WINDOW_NAME, 
        LPCWSTR argNameApiClass = DEF_WINDOW_CLASS_NAME, 
        int argWidth = DEF_WINDOW_WIDTH, 
        int argHeight = DEF_WINDOW_HEIGHT, 
        int argDpi = DEF_WINDOW_DPI,
        double argMaxFps = DEF_MAX_FPS,
        UINT argWndStyle = DEF_WINDOW_STYLE,
        LPWSTR argLoadIcon = DEF_WINDOW_LOAD_ICON, 
        LPWSTR argLoadCursor = DEF_WINDOW_LOAD_CURSOR,
        int argBackgroundColor = DEF_WINDOW_BACKGROUND_COLOR,
        LPWSTR argSmallIcon = DEF_WINDOW_SMALL_ICON,
        bool argMinimizeAuto = false,
        bool argSingleExistence = false
    ) : name(argName), 
        nameApiClass(argNameApiClass), 
        width(argWidth), 
        height(argHeight), 
        dpi(argDpi), 
        style(argWndStyle), 
        loadIcon(argLoadIcon), 
        loadCursor(argLoadCursor),
        backgroundColor(argBackgroundColor), 
        smallIcon(argSmallIcon),
        minimizeAuto(argMinimizeAuto),
        singleExistence(argSingleExistence),
        lpPixel(nullptr) {
        fps.max = argMaxFps;
        lpPixel = (LPDWORD)HeapAlloc(
        GetProcessHeap(),
        HEAP_ZERO_MEMORY,
        width *height * 4);
    }

    void getFps();
    void updateMaxFps();
    void updateSize();
    // void copyArgBuffer();

    void create(HINSTANCE arg_hinstance, WINDOW_PROC_TYPE* pt_window_proc);
    void updateStatus(int arg_status);
    bool isVisible();
    void graphicLoop();

    bool minimizeMsg(HWND arg_hwnd);
    bool killFocusMsg(HWND arg_hwnd, bool single_window);
    bool setFocusMsg(HWND arg_hwnd);
    bool sizeMsg(HWND arg_hwnd, LPARAM l_param);
    bool createMsg(HWND arg_hwnd);
    bool closeMsg(HWND arg_hwnd);
    bool destroyMsg(HWND arg_hwnd);
    bool paintMsg(HWND arg_hwnd);
    bool userMsg(HWND arg_hwnd);

private :
    int status = WINDOW_STATUS_DEF;
    bool focus = true;
    bool created = false;
    bool visible = true;
    bool minimizeAuto;
    bool singleExistence;

    LPCWSTR name;
    LPCWSTR nameApiClass;

    int width;
    int height;
    int dpi;

    UINT style;
    LPWSTR loadIcon;
    LPWSTR loadCursor;
    int backgroundColor;
    LPWSTR smallIcon;

    HWND hWnd;

    WNDCLASSEX wndClass;

    Fps fps;

    HDC hWndDC = nullptr;
    PAINTSTRUCT hPs;

    HDC hBufDC = nullptr;
    HBITMAP hBufBmp = nullptr;
    BITMAPINFO hBufBmpInfo;
    LPDWORD lpPixel;


};

#endif WINDOW_H_