#include "window.h"

LRESULT __stdcall Window::procedure(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
        switch (msg)
        {
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
