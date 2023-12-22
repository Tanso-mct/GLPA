#include "user_input.h"


void UserInput::add(
    std::wstring funcName, 
    GLPA_USER_FUNC ptAddFunc,
    HWND getMsgWndHwnd, 
    int msgType
){
    switch (msgType)
    {
    case GLPA_USERINPUT_MESSAGE_KEYDOWN :
        myFunc[funcName] = ptAddFunc;
        msgFunc[funcName].push_back(GLPA_USERINPUT_MESSAGE_KEYDOWN);
        keyDownFunc[getMsgWndHwnd].push_back(funcName);
        break;
    
    default:
        throw std::runtime_error(ERROR_USER_INPUT_ADD);
        break;
    }
}


void UserInput::edit(std::wstring funcName, GLPA_USER_FUNC editedFunc){
    myFunc[funcName] = editedFunc;
}


void UserInput::release(std::wstring funcName){
    myFunc.erase(funcName);

    std::unordered_map<HWND, std::vector<int>> eraseIndex;
    for (auto loopMsg : msgFunc[funcName]){
        switch (loopMsg)
        {
        case GLPA_USERINPUT_MESSAGE_KEYDOWN:
            for (auto& loopKeyDownFunc : keyDownFunc){
                for (int vecIndex = 0; vecIndex < loopKeyDownFunc.second.size(); vecIndex++){
                    if (loopKeyDownFunc.second[vecIndex] == funcName){
                        eraseIndex[loopKeyDownFunc.first].push_back(vecIndex);
                    }
                }
            }

            for (auto it : eraseIndex){
                for (auto index : it.second){
                    keyDownFunc[it.first].erase(keyDownFunc[it.first].begin() + index);
                }
            }

            
            break;
        
        default:
            break;
        }
    }
}


void UserInput::keyDown(HWND hWnd, std::string scName, UINT msg, WPARAM wParam, LPARAM lParam)
{
    for (auto funcName : keyDownFunc[hWnd]){
        myFunc[funcName](scName, msg, wParam, lParam);
    }
}
