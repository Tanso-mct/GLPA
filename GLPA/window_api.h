#ifndef WINDOWAPI_H_
#define WINDOWAPI_H_

#include <Windows.h>
#include <string>
#include <unordered_map>
#include <tchar.h>
#include <functional>

#include "window.h"
#include "user_input.h"

class WindowApi
{
public :
    /// @brief Register a class using the window api for a specific window.
    /// @param window_name The registration name of the window in which the class registration is to be made.
    /// @param window Map variable to hold all windows.
    void createWindow(LPCWSTR window_name, std::unordered_map<LPCWSTR, Window>* window);

    void showWindow(LPCWSTR window_name, std::unordered_map<LPCWSTR, Window>* window);

    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;
    WINDOW_PROC_TYPE* ptWindowProc;

private :
    UserInput userInput;
};


#endif WINDOWAPI_H_

