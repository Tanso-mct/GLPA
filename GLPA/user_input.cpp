#include "user_input.h"

void UserInput::add(
    std::wstring funcName, 
    userFunc ptAddFunc,
    HWND getMsgWndHwnd, 
    int msgType
){
    switch (msgType)
    {
    case GLPA_USERINPUT_MESSAGE_KEYDOWN :
        myFunc.emplace(funcName, ptAddFunc);
        myFunc[funcName] = ptAddFunc;
        keyDownFunc[getMsgWndHwnd].push_back(funcName);
        break;
    
    default:
        throw std::runtime_error(ERROR_USER_INPUT_ADD);
        break;
    }
}

void UserInput::typingStart(){
    typing = true;
}

void UserInput::typingEnd(){
    typing = false;
}

void UserInput::keyDown(HWND hWnd, std::string scName, WPARAM wParam, LPARAM lParam)
{
    for (auto funcName : keyDownFunc[hWnd]){
        myFunc[funcName](scName, wParam, lParam);
    }
}
