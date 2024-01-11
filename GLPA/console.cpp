#include "console.h"

Console::Console(Scene2d* argPtScene2d){
    scene = argPtScene2d;
}


std::wstring Console::getCommandName(std::wstring inputText){
    int tag = inputText.find(COMMAND_TAG_C);
    std::wstring rtCommandName = inputText.substr(tag + COMMAND_TAG_C_SIZE, inputText.size() - 1);
    return rtCommandName;
}


void Console::mouseLbtnDown(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    selectingTextGroup = scene->text.getGroupOnMouse(lParam, scene->useWndDpi);

    if (selectingTextGroup != GLPA_NULL_WTEXT){
        glpa.userInput.typing = true;
    }
}


void Console::keyDown(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    if (selectingTextGroup == GLPA_NULL_WTEXT){
        return;
    }

    std::wstring inputLowWstr = glpa.userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && glpa.userInput.typing){
        glpa.userInput.typingDownScene2d(scene, selectingTextGroup, wParam, inputLowWstr);
    }

    switch (wParam){
    case VK_RETURN:
        if (glpa.userInput.typing){
            command.execute(
                getCommandName(
                    scene->text.typingMarkDelete(
                        scene->text.getGroupLastLineWstr(selectingTextGroup)
                    )
                )
            );
            if (scene->text.getGroupLineAmount(selectingTextGroup) >= 22){
                textStartLine += 1;
                scene->text.setStartLine(selectingTextGroup, textStartLine);
            }
            glpa.userInput.typingNewLineScene2d(scene, selectingTextGroup, L"<console>");

            
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

    std::wstring inputLowWstr = glpa.userInput.convertWParamToLowWstr(wParam);

    if (scName == SCENE_GLPA_CONSOLE && glpa.userInput.typing){
        glpa.userInput.typingUpScene2d(scene, selectingTextGroup, wParam, inputLowWstr);
    }
}


void Console::mainUpdate(HDC hBufDC, LPDWORD lpPixel){
    if (selectingTextGroup != GLPA_NULL_WTEXT){
        scene->text.typingMarkAnime(&(glpa.userInput.typing), &(scene->edited), selectingTextGroup);
    }
    
}
