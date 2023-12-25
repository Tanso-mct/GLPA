#ifndef CONSOLE_H_
#define CONSOLE_H_

#include <string>
#include <Windows.h>
#include <tchar.h>
#include <sstream>
#include <chrono>

#include "glpa.h"

#define WINDOW_CONSOLE L"Console"
#define WINDOWCLASS_CONSOLE L"window_console"

#define SCENE_GLPA_CONSOLE "glpa_console"
#define SCENE_FOLDER_PASS_GLPA_CONSOLE L"resource/scene/glpa_console"

class Console
{
public :
    Console(Glpa *argPtGlpa);

    void setScenePt(Scene2d *arg_pt_scene_2d);

    GLPA_USER_FUNC(tempTypingDown);
    GLPA_USER_FUNC(tempTypingUp);

    GLPA_SCENE_FUNC(tempSceneLoop);

private :
    Glpa* ptGlpa;
    Scene2d* ptScene2d;

    double tempFrameCount = 0;
    bool sceneStart = false;

    bool turnOn = false;

    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    std::chrono::milliseconds duration;
};


#endif CONSOLE_H_

