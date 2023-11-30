#ifndef WINDOWAPI_H_
#define WINDOWAPI_H_

#include <Windows.h>
#include <string>
#include <unordered_map>
#include <tchar.h>

#include "window.h"
#include "user_input.h"

class WindowApi
{
public :
    /// @brief Convert a variable of type String to a variable of type lpcwstr.
    /// @param str Variables of type string to be converted.
    /// @return Variables of type lpcwstr that have been converted.
    LPCWSTR convertStringToLPCWSTR(std::string str);

    /// @brief Register a class using the window api for a specific window.
    /// @param window_name The registration name of the window in which the class registration is to be made.
    /// @param window Map variable to hold all windows.
    void registerClass(std::string window_name, std::unordered_map<std::string, Window> window);

    /// @brief The window is created using the window api function.
    /// @param window_name The registered name of the window to be created.
    /// @param window Map variable to hold all windows.
    void createWindow(std::string window_name, std::unordered_map<std::string, Window> window);

    void getWindowMessage(std::string window_name, std::unordered_map<std::string, Window> window);

    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;

private :
    UserInput userInput;
};


#endif WINDOWAPI_H_

