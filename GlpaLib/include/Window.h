#ifndef GLPA_WINDOW_H_
#define GLPA_WINDOW_H_

#include <Windows.h>
#include <stdexcept>

#include <d2d1.h>
#pragma comment(lib, "d2d1")

namespace Glpa 
{

class Window
{
private :
    LPCWSTR name = L"TEMP_NAME";
    LPCWSTR apiClassName = L"TEMP_API_CLASS_NAME";

    int width = 1200;
    int height = 800;
    int dpi = 1;

    UINT style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
    LPWSTR loadIcon = IDI_APPLICATION;
    LPWSTR loadCursor = IDC_ARROW;
    int bgColor = WHITE_BRUSH;
    LPWSTR smallIcon = IDI_APPLICATION;

    DWORD viewStyle = WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;

    HDC hWndDC = nullptr;
    PAINTSTRUCT hPs;

    HBITMAP hBufBmp = nullptr;
    BITMAPINFO hBufBmpInfo;

    ID2D1Factory* pFactory = NULL;
    ID2D1HwndRenderTarget* pRenderTarget = NULL;
    ID2D1Bitmap* pBitmap = NULL;

public :
    ~Window();
    
    HWND hWnd = nullptr;
    WNDCLASSEX apiClass;

    HDC hBufDC = nullptr;
    LPDWORD pixels;
    
    void createPixels();
    void create(HINSTANCE hInstance);

    /// @brief Create a device context.
    void createDc();

    /// @brief Called when redrawing is required and copies the pixels data to the buffer.
    void paint();

    void SetName(LPCWSTR str) {name = str;}
    void SetApiClassName(LPCWSTR str) {apiClassName = str;}

    int GetWidth() const {return width;}
    void SetWidth(int value) {width = value;}

    int GetHeight() const {return height;}
    void SetHeight(int value) {height = value;}

    int GetDpi() const {return dpi;}
    void setDpi(int value) {dpi = value;}

    void setViewStyle(UINT value);
    void addViewStyle(UINT value);
    void deleteViewStyle(UINT value);

    void setLoadIcon(LPWSTR value) {loadIcon = value;}
    void setLoadCursor(LPWSTR value) {loadCursor = value;}
    void setBgColor(int value) {bgColor = value;}
    void setSmallIcon(LPWSTR value) {smallIcon = value;}

    /// @brief Send a redraw message.
    void sendPaintMsg();
};

}

#endif GLPA_WINDOW_H_
