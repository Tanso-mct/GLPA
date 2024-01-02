#ifndef CONSOLE_H_
#define CONSOLE_H_

#include <string>
#include <Windows.h>
#include <tchar.h>
#include <sstream>
#include <chrono>
#include <functional>
#include <unordered_map>

#include "glpa.h"

#include "command.h"

#define WINDOW_CONSOLE L"Console"
#define WINDOWCLASS_CONSOLE L"window_console"

#define SCENE_GLPA_CONSOLE "glpa_console"
#define SCENE_FOLDER_PASS_GLPA_CONSOLE L"resource/scene/glpa_console"

class Console
{
public :
    Console(Glpa *argPtGlpa, Scene2d* argPtScene2d);

    std::wstring getCommandName(std::wstring input_text);

    GLPA_USER_FUNC(mouseLbtnDown);

    GLPA_USER_FUNC(keyDown);
    GLPA_USER_FUNC(keyUp);

    GLPA_SCENE_FUNC(mainUpdate);

private :
    Glpa* ptGlpa;
    Scene2d* ptScene2d;

    Command command;

    std::wstring selectingTextGroup = GLPA_NULL_WTEXT;

    int textStartLine = 0;
    std::unordered_map<std::wstring, int> lastFrameTextSize;

};


#endif CONSOLE_H_

