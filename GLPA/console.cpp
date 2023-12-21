#include "console.h"

void Console::setScenePt(Scene2d *argPtScene2d){
    ptScene2d = argPtScene2d;
}

void Console::tempTyping(std::string scName, WPARAM wParam, LPARAM lParam){
    std::wstring tempStr;
    if (scName == SCENE_GLPA_CONSOLE){
        switch (wParam)
        {
        case VK_RETURN:
            tempStr = ptScene2d->text.getGroupLastLineWstr(L"Temp");
            ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, tempStr + L" EDITED");
            ptScene2d->edited = true;
            break;
        
        default:
            break;
        }
    }
}