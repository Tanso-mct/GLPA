#include "user_input.h"


void UserInput::add(
    std::wstring funcName, 
    GLPA_USER_FUNC_FUNCTIONAL ptAddFunc,
    HWND getMsgWndHwnd, 
    int msgType
){
    switch (msgType){
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_KEYDOWN, keyDownFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_KEYUP, keyUpFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMOVE, mouseMoveFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNDOWN, mouseLbtnDownFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNUP, mouseLbtnUpFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNDBCLICK, mouseLbtnDblclickFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNDOWN, mouseRbtnDownFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNUP, mouseRbtnUpFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNDBCLICK, mouseRbtnDblClickFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNDOWN, mouseMbtnDownFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNUP, mouseMbtnUpFunc)
    GLPA_USERINPUT_ADD_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNDBWHEEL, mouseMbtnWheelFunc)

    default:
        throw std::runtime_error(ERROR_USER_INPUT_ADD);
        break;
    }
}


void UserInput::edit(std::wstring funcName, GLPA_USER_FUNC_FUNCTIONAL editedFunc){
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
        switch (loopMsg){
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_KEYDOWN, keyDownFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_KEYUP, keyUpFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMOVE, mouseMoveFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNDOWN, mouseLbtnDownFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNUP, mouseLbtnUpFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSELBTNDBCLICK, mouseLbtnDblclickFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNDOWN, mouseRbtnDownFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNUP, mouseRbtnUpFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSERBTNDBCLICK, mouseRbtnDblClickFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNDOWN, mouseMbtnDownFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNUP, mouseMbtnUpFunc)
        GLPA_USERINPUT_RELEASE_CASE(GLPA_USERINPUT_MESSAGE_MOUSEMBTNDBWHEEL, mouseMbtnWheelFunc)
        
        default:
            break;
        }
    }
}


std::wstring UserInput::convertWParamToLowWstr(WPARAM wParam){
    wchar_t wParamChar = static_cast<wchar_t>(wParam);
    std::wstring upperWstr = std::wstring(1, wParamChar);
    LPWSTR lpwStr = const_cast<LPWSTR>(upperWstr.c_str());
    std::wstring rtLowWstr = CharLower(lpwStr);
    return rtLowWstr;
}


GLPA_USERINPUT_MSG_FUNC_DEFINE(keyDown, keyDownFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(keyUp, keyUpFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseMove, mouseMoveFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseLbtnDown, mouseLbtnDownFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseLbtnUp, mouseLbtnUpFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseLbtnDblclick, mouseLbtnDblclickFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseRbtnDown, mouseRbtnDownFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseRbtnUp, mouseRbtnUpFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseRbtnDblClick, mouseRbtnDblClickFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseMbtnDown, mouseMbtnDownFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseMbtnUp, mouseMbtnUpFunc)
GLPA_USERINPUT_MSG_FUNC_DEFINE(mouseMbtnWheel, mouseMbtnWheelFunc)


void UserInput::typingNewLineScene2d(Scene2d *ptScene2d, std::wstring textGroupName, std::wstring addLineText){
    std::wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(textGroupName);

    if (lastLineWstr.size() != 0){
        if (lastLineWstr.back() == GLPA_TYPING_MARK){
            ptScene2d->text.edit(
                textGroupName, 
                GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr.substr(0, lastLineWstr.size() - 1)
            );
        }
    }

    ptScene2d->text.addText(textGroupName, addLineText);
    ptScene2d->edited = true;
}


void UserInput::typingFinishScene2d(Scene2d *ptScene2d, std::wstring textGroupName){
    std::wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(textGroupName);
    typing = false;

    if (lastLineWstr.size() != 0){
        if (lastLineWstr.back() == GLPA_TYPING_MARK){
            ptScene2d->text.edit(
                textGroupName, 
                GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr.substr(0, lastLineWstr.size() - 1)
            );
        }
    }

    ptScene2d->edited = true;
}


void UserInput::typingDownScene2d(
    Scene2d* ptScene2d, std::wstring textGroupName, WPARAM wParam, std::wstring wParamWstr
){
    std::wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(textGroupName);
    std::wstring editedLineWstr;

    switch (wParam){
    case VK_BACK:
        ptScene2d->text.typingDelete(textGroupName);

        ptScene2d->edited = true;
        break;

    case VK_SHIFT:
        shift = true;
        break;

    case VK_SPACE:
        ptScene2d->text.typingAdd(textGroupName, L" ");

        ptScene2d->edited = true;
        break;

    case VK_OEM_2:
        ptScene2d->text.typingAdd(textGroupName, L"/");

        ptScene2d->edited = true;
        break;
    
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
    case 'T':
    case 'U':
    case 'V':
    case 'W':
    case 'X':
    case 'Y':
    case 'Z':
        if (shift){
            ptScene2d->text.typingAdd(textGroupName, std::wstring(1, wParam));
        }
        else{
            ptScene2d->text.typingAdd(textGroupName, wParamWstr);
        }
        
        ptScene2d->edited = true;
        break;
    }

}


void UserInput::typingUpScene2d(
    Scene2d* ptScene2d, std::wstring textGroupName, WPARAM wParam, std::wstring wParamWstr
){
    std::wstring tempStr;
    switch (wParam){
    case VK_SHIFT:
        shift = false;
        break;
    }
    
}
