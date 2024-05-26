#include "Window.h"

Glpa::Window::Window
(
    LPCWSTR argName, LPCWSTR argApiClassName, int argWidth, int argHeight, int argDpi, 
    UINT argStyle, LPWSTR argLoadIcon, LPWSTR argLoadCursor, int argBackgroundColor, LPWSTR argSmallIcon
){
    name = argName;
    apiClassName = argApiClassName;

    width = argWidth;
    height = argHeight;
    dpi = argDpi;

    style = argStyle;
    loadIcon = argLoadIcon;
    loadCursor = argLoadCursor;
    bgColor = argBackgroundColor;
    smallIcon = argSmallIcon;

    pixels = (LPDWORD)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, width *height * 4);
}

void Glpa::Window::sendPaintMsg()
{
    InvalidateRect(hWnd, NULL, FALSE);
}
