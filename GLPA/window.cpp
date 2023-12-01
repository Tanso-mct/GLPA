#include "window.h"

LRESULT CALLBACK Window::procedure(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
        switch (msg)
        {
        case WM_CREATE :
                return 0;

        case WM_PAINT :
                return 0;

        case WM_CLOSE :
                DestroyWindow(hWnd);

        case WM_DESTROY :
                PostQuitMessage(0);

                return 0;

        default :
                return DefWindowProc(hWnd, msg, wParam, lParam);
        }
        return 0;
}
