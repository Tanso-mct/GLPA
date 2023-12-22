#include "console.h"

Console::Console(Glpa *argPtGlpa){
    ptGlpa = argPtGlpa;
}


void Console::setScenePt(Scene2d *argPtScene2d){
    ptScene2d = argPtScene2d;
}


void Console::tempTyping(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    wchar_t wParamChar = static_cast<wchar_t>(wParam);
    std::wstring keyWstr = std::wstring(1, wParamChar);
    LPWSTR lpwStr = const_cast<LPWSTR>(keyWstr.c_str());
    std::wstring inputChar = CharLower(lpwStr);

    if (scName == SCENE_GLPA_CONSOLE && !ptGlpa->userInput.typing){
        switch (wParam){
        case 0x0D:
            ptScene2d->text.setStartLine(L"Temp", 2);
            ptScene2d->edited = true;
            glpa.userInput.typing = true;
            break;
        
        default:
            break;
        }
        
    }
    else if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        switch (wParam){
        case VK_RETURN:
            ptScene2d->edited = true;
            glpa.userInput.typing = false;
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
}