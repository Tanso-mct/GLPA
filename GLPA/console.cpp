#include "console.h"

Console::Console(Glpa *argPtGlpa, Scene2d* argPtScene2d){
    ptGlpa = argPtGlpa;
    ptScene2d = argPtScene2d;
}


std::wstring Console::getCommandName(std::wstring inputText){
    int tag = inputText.find(COMMAND_TAG_C);
    std::wstring rtCommandName = inputText.substr(tag + COMMAND_TAG_C_SIZE, inputText.size() - 1);
    return rtCommandName;
}


void Console::mouseLbtnDown(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    selectingTextGroup = ptScene2d->text.getGroupOnMouse(lParam, ptScene2d->useWndDpi);

    if (selectingTextGroup != GLPA_NULL_WTEXT){
        glpa.userInput.typing = true;
    }
}


void Console::keyDown(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    if (selectingTextGroup == GLPA_NULL_WTEXT){
        return;
    }

    std::wstring inputLowWstr = ptGlpa->userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        ptGlpa->userInput.typingDownScene2d(ptScene2d, selectingTextGroup, wParam, inputLowWstr);
    }

    switch (wParam){
    case VK_RETURN:
        if (glpa.userInput.typing){
            command.execute(
                getCommandName(
                    ptScene2d->text.typingMarkDelete(
                        ptScene2d->text.getGroupLastLineWstr(selectingTextGroup)
                    )
                )
            );
            if (ptScene2d->text.getGroupLineAmount(selectingTextGroup) >= 22){
                textStartLine += 1;
                ptScene2d->text.setStartLine(selectingTextGroup, textStartLine);
            }
            ptGlpa->userInput.typingNewLineScene2d(ptScene2d, selectingTextGroup, L"<console>");

            
        }
        else {
            glpa.userInput.typing = true;
        }
        break;
    
    default:
        break;
    }

}


void Console::keyUp(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    if (selectingTextGroup == GLPA_NULL_WTEXT){
        return;
    }

    std::wstring inputLowWstr = ptGlpa->userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && ptGlpa->userInput.typing){
        ptGlpa->userInput.typingUpScene2d(ptScene2d, selectingTextGroup, wParam, inputLowWstr);
    }
}


void Console::mainUpdate(HDC hBufDC, LPDWORD lpPixel){
    if (selectingTextGroup != GLPA_NULL_WTEXT){
        ptScene2d->text.typingMarkAnime(&(ptGlpa->userInput.typing), &(ptScene2d->edited), selectingTextGroup);
    }
    
}
