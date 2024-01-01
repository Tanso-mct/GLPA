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

    if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        ptGlpa->userInput.keyUpTypingScene2d(ptScene2d, L"Temp", wParam, inputLowWstr);
    }
}


GLPA_SCENE_FUNC_DEFINE(Console, tempSceneLoop, hBufDC, lpPixel, width, height, dpi){
    std:: wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(selectingTextGroup);
    int thisFrameTextSize = (ptScene2d->text.getGroupLastLineWstr(selectingTextGroup)).size();

    if (lastFrameTextSize.find(selectingTextGroup) == lastFrameTextSize.end()){
        startTime = std::chrono::high_resolution_clock::now();
    }
    else if (lastFrameTextSize[selectingTextGroup] != thisFrameTextSize){
        startTime = std::chrono::high_resolution_clock::now();
    }

    endTime = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (ptGlpa->userInput.typing){
        if (duration.count() >= 0 && duration.count() < 500){
            if (!turnOn){
                ptScene2d->text.edit(selectingTextGroup, GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr + GLPA_TYPING_MARK);
                ptScene2d->edited = true;
                
                turnOn = true;
            }
        }
        else if (duration.count() < 1000){
            if (lastLineWstr.size() != 0){
                if (lastLineWstr.back() == GLPA_TYPING_MARK){
                    ptScene2d->text.edit(selectingTextGroup, GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr.substr(0, lastLineWstr.size() - 1));
                    ptScene2d->edited = true;

                    turnOn = false;
                }
            }
        }
        else{
            startTime = endTime;
        }
    }

    lastFrameTextSize[selectingTextGroup] = (ptScene2d->text.getGroupLastLineWstr(selectingTextGroup)).size();
    
}
