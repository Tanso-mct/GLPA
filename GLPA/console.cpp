#include "console.h"

Console::Console(Glpa *argPtGlpa){
    ptGlpa = argPtGlpa;
}


void Console::setScenePt(Scene2d *argPtScene2d){
    ptScene2d = argPtScene2d;
}


void Console::tempTyping(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    TCHAR pressedChar = static_cast<TCHAR>(wParam);
    std::wstring keyWstr = std::wstring(1, pressedChar);

    std::wstring tempStr;
    if (scName == SCENE_GLPA_CONSOLE && !ptGlpa->userInput.typing){
        switch (wParam){
        case VK_RETURN:
            tempStr = ptScene2d->text.getGroupLastLineWstr(L"Temp");
            ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, tempStr + L" EDITED");
            ptScene2d->edited = true;
            glpa.userInput.typing = true;
            break;
        
        default:
            break;
        }
        
    }
    else if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        // switch (wParam){
        // case /* */:
            
        //     break;
        
        // default:
        //     break;
        // }
    }
}