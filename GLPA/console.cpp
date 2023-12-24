#include "console.h"

Console::Console(Glpa *argPtGlpa){
    ptGlpa = argPtGlpa;
}


void Console::setScenePt(Scene2d *argPtScene2d){
    ptScene2d = argPtScene2d;
}


GLPA_USER_FUNC_DEFINE(Console, tempTypingDown, scName, msg, wParam, lParam){
    std::wstring inputLowWstr = ptGlpa->userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && !ptGlpa->userInput.typing){
        switch (wParam){
        case VK_RETURN:
            ptScene2d->text.setStartLine(L"Temp", 2);
            ptScene2d->edited = true;
            glpa.userInput.typing = true;
            break;
        
        default:
            break;
        }
        
    }
    else if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        ptGlpa->userInput.keyDownTypingScene2d(ptScene2d, L"Temp", wParam, inputLowWstr);
    }
}


GLPA_USER_FUNC_DEFINE(Console, tempTypingUp, scName, msg, wParam, lParam){
    std::wstring inputLowWstr = ptGlpa->userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && !ptGlpa->userInput.typing){
        switch (wParam){
        case VK_RETURN:
            
            break;
        
        default:
            break;
        }
        
    }
    else if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        ptGlpa->userInput.keyUpTypingScene2d(ptScene2d, L"Temp", wParam, inputLowWstr);
    }
}
