#ifndef GLPA_WINDOW_H_
#define GLPA_WINDOW_H_

#include <Windows.h>
#include <string>
#include <locale>
#include <codecvt>

#include <d2d1.h>
#pragma comment(lib, "d2d1")

#include "ErrorHandler.h"

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

    ID2D1Factory* pFactory = nullptr;

    std::wstring_convert<std::codecvt_utf8<wchar_t>> strConverter;

public :
    Window();
    ~Window();

    ID2D1HwndRenderTarget* pRenderTarget = nullptr;
    ID2D1Bitmap* pBitmap = nullptr;
    
    HWND hWnd = nullptr;
    PAINTSTRUCT hPs;
    WNDCLASSEX apiClass;

    LPDWORD pixels;

    bool isFocusing = false;
    
    void create(HINSTANCE hInstance);

    /// @brief Create a D2D.
    void initD2D();

    void releaseD2D();

    void SetName(LPCWSTR str) {name = str;}
    void SetApiClassName(LPCWSTR str) {apiClassName = str;}

    int GetWidth() const {return width;}
    void SetWidth(int value);

    int GetHeight() const {return height;}
    void SetHeight(int value);

    int GetDpi() const {return dpi;}
    void SetDpi(int value);

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
