#ifndef WINDOW_H_
#define WINDOW_H_

#include <Windows.h>
#include <string>

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

class Window
{
public :
    Window
    (
        LPCWSTR argName = DEF_WINDOW_NAME, 
        LPCWSTR argNameApiClass = DEF_WINDOW_CLASS_NAME, 
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

        // WNDCLASSEX wndClass;
        // wndClass.cbSize = sizeof(wndClass);
        // wndClass.style = window[wndName].style;
        // wndClass.lpfnWndProc = window[wndName].procedure;
        // wndClass.cbClsExtra = NULL;
        // wndClass.cbWndExtra = NULL;
        // wndClass.hInstance = hInstance;
        // wndClass.hIcon = (HICON)LoadImage
        // (
        //     NULL, 
        //     MAKEINTRESOURCE(window[wndName].loadIcon),
        //     IMAGE_ICON,
        //     0,
        //     0,
        //     LR_DEFAULTSIZE | LR_SHARED
        // );
        // wndClass.hCursor = (HCURSOR)LoadImage
        // (
        //     NULL, 
        //     MAKEINTRESOURCE(window[wndName].loadCursor),
        //     IMAGE_CURSOR,
        //     0,
        //     0,
        //     LR_DEFAULTSIZE | LR_SHARED
        // );                                                 
        // wndClass.hbrBackground = (HBRUSH)GetStockObject(window[wndName].backgroundColor);
        // wndClass.lpszMenuName = NULL;
        // wndClass.lpszClassName = window[wndName].nameApiClass;
        // wndClass.hIconSm =
        // LoadIcon(wndClass.hInstance, MAKEINTRESOURCE(window[wndName].smallIcon));

        // if (!RegisterClassEx(&wndClass))
        // {
        //     MessageBox(
        //         NULL,
        //         _T("RegisterClassEx fail"),
        //         wndName,
        //         MB_ICONEXCLAMATION
        //     );
        // }
    }
    
    // MyWindow(HINSTANCE hInstance, LPCWSTR title, int width, int height)
    //     : hInstance(hInstance), title(title), width(width), height(height) {
    //     // ウィンドウクラスの登録
    //     WNDCLASS wc = { 0 };
    //     wc.lpfnWndProc = WindowProc;
    //     wc.hInstance = hInstance;
    //     wc.lpszClassName = L"MyWindowClass";

    //     if (!RegisterClass(&wc)) {
    //         // 登録に失敗した場合のエラーハンドリング
    //         throw std::runtime_error("Failed to register window class");
    //     }

    //     // ウィンドウの作成
    //     hwnd = CreateWindow(
    //         L"MyWindowClass", title, WS_OVERLAPPEDWINDOW,
    //         CW_USEDEFAULT, CW_USEDEFAULT, width, height,
    //         NULL, NULL, hInstance, this);

    //     if (!hwnd) {
    //         // ウィンドウの作成に失敗した場合のエラーハンドリング
    //         throw std::runtime_error("Failed to create window");
    //     }
    // }

    // double getFps();
    // void changeSize();
    // void copyArgBuffer();
    static LRESULT CALLBACK procedure(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

    bool existence = false;
    bool foucus = false;

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