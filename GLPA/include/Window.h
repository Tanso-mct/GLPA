#ifndef GLPA_WINDOW_H_
#define GLPA_WINDOW_H_

#include <Windows.h>
#include <stdexcept>

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

    UINT style = CS_HREDRAW | CS_VREDRAW;
    LPWSTR loadIcon = IDI_APPLICATION;
    LPWSTR loadCursor = IDC_ARROW;
    int bgColor = WHITE_BRUSH;
    LPWSTR smallIcon = IDI_APPLICATION;

    DWORD viewStyle = WS_SYSMENU;

    HDC hWndDC = nullptr;
    PAINTSTRUCT hPs;

    HDC hBufDC = nullptr;
    HBITMAP hBufBmp = nullptr;
    BITMAPINFO hBufBmpInfo;

    LPDWORD pixels;

public :
    HWND hWnd;
    WNDCLASSEX apiClass;
    void createPixels();
    void create(HINSTANCE hInstance);

    void createDc();
    void paint();

    void setName(LPCWSTR str) {name = str;}
    void setApiClassName(LPCWSTR str) {apiClassName = str;}

    int getWidth() const {return width;}
    void setWidth(int value) {width = value;}

    int getHeight() const {return height;}
    void setHeight(int value) {height = value;}

    void setDpi(int value) {dpi = value;}

    void setStyle(UINT value) {style = value;}
    void setLoadIcon(LPWSTR value) {loadIcon = value;}
    void setLoadCursor(LPWSTR value) {loadCursor = value;}
    void setBgColor(int value) {bgColor = value;}
    void setSmallIcon(LPWSTR value) {smallIcon = value;}

    void sendPaintMsg();
};

}

#endif GLPA_WINDOW_H_
