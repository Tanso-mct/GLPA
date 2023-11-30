#ifndef WINDOWAPI_H_
#define WINDOWAPI_H_

#include <Windows.h>
#include <string>

#include "window.h"
#include "user_input.h"

class WindowApi
{
public :
    void storeWinMainArgument
    (
        _In_ HINSTANCE arg_hInstance, _In_opt_ HINSTANCE arg_hPrevInstance, 
        _In_ LPSTR arg_lpCmdLine, _In_ int _nCmdShow
    );

    void registerClass(Window window);

    void createWindow(Window window);

    void getWindowMessage(Window window);

private :
    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;
    
    UserInput userInput;
};


#endif WINDOWAPI_H_

