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


void UserInput::eraseFunc(std::wstring funcName, std::unordered_map<HWND, std::vector<std::wstring>>* ptMsgFunc){
    std::unordered_map<HWND, std::vector<int>> eraseIndex;
    for (auto& loopMsgFunc : *ptMsgFunc){
        for (int vecIndex = 0; vecIndex < loopMsgFunc.second.size(); vecIndex++){
            if (loopMsgFunc.second[vecIndex] == funcName){
                eraseIndex[loopMsgFunc.first].push_back(vecIndex);
            }
        }
    }

    for (auto it : eraseIndex){
        for (auto index : it.second){
            (*ptMsgFunc)[it.first].erase((*ptMsgFunc)[it.first].begin() + index);
        }
    }
}


void UserInput::release(std::wstring funcName)
{
    myFunc.erase(funcName);
    for (auto loopMsg : msgFunc[funcName]){
        switch (loopMsg)
        {
        case GLPA_USERINPUT_MESSAGE_KEYDOWN:
            eraseFunc(funcName, &keyDownFunc);
            
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


void UserInput::keyTypingScene2d(Scene2d* ptScene2d, WPARAM wParam){
    std::wstring tempStr;
    switch (wParam){
    case VK_RETURN:
        ptScene2d->edited = true;
        typing = false;
        break;

    case VK_OEM_2:
        tempStr = ptScene2d->text.getGroupLastLineWstr(L"Temp");
        ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, tempStr + L"/");
        ptScene2d->edited = true;
        break;
    
    default:
        tempStr = ptScene2d->text.getGroupLastLineWstr(L"Temp");
        ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, tempStr + inputChar);
        ptScene2d->edited = true;
        break;
    }
}
