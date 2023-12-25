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


GLPA_SCENE_FUNC_DEFINE(Console, tempSceneLoop, hBufDC, lpPixel, width, height, dpi){
    // tempFrameCount += 0.1;
    // std::wstringstream wss;
    // wss << tempFrameCount;

    // // std::wstring‚É•ÏŠ·
    // std::wstring tempConvertedWString = wss.str();

    // std::wstringstream wss;
    // wss << duration.count();
    // std::wstring wstrDuration = wss.str();


    // if (ptGlpa->userInput.typing){
    //     std:: wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(L"Temp");

    //     if (lastLineWstr.back() == L'|') {
    //         ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr.substr(0, lastLineWstr.size() - 1));
    //         ptScene2d->edited = true;
    //         return;
    //     }
    // }

    if (!sceneStart){
        startTime = std::chrono::high_resolution_clock::now();
        sceneStart = true;
    }

    endTime = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std:: wstring lastLineWstr = ptScene2d->text.getGroupLastLineWstr(L"Temp");
    if (duration.count() >= 500){
        if (!turnOn && ptGlpa->userInput.typing){
            ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr + GLPA_TYPING_MARK);
            ptScene2d->edited = true;
            
            turnOn = true;
            startTime = endTime;
        }
        else if (turnOn && ptGlpa->userInput.typing){
            if (lastLineWstr.size() != 0){
                if (lastLineWstr.back() == GLPA_TYPING_MARK){
                    ptScene2d->text.edit(L"Temp", GLPA_TEXT_EDIT_GROUP_LAST, lastLineWstr.substr(0, lastLineWstr.size() - 1));
                    ptScene2d->edited = true;

                    turnOn = false;
                    startTime = endTime;
                }
            }
        }
    }

}
