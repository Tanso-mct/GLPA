#ifndef WINDOWAPI_H_
#define WINDOWAPI_H_

#include <string>

#include "window.h"
#include "user_input.h"

class WindowApi
{
    public :

    void registerClass(Window window);

    void createWindow(Window window);

    void getWindowMessage(Window window);

    private :
    UserInput userInput;
};


#endif WINDOWAPI_H_

