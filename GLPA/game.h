#ifndef GAME_H_
#define GAME_H_

#include "glpa.h"

#include "command.h"
#include "scene3d.h"
#include "user_input.h"


#define WINDOW_GAME L"Game"
#define WINDOW_CLASS_GAME L"window_game"
#define SCENE_FOLDER_PASS_GAME L"resource/scene/game_1"


#define SCENE_GAME_1 "scene_game_1"

class Game{
public :

    void createWnd();
    void createScene();
    void loadScene();
    void selectScene();

    void tempStart();
    void tempRelease();

    // void camReset();

    std::function<void(std::string, UINT, WPARAM, LPARAM)> keyDownFunc;

    GLPA_USER_FUNC(keyDown);
    // GLPA_USER_FUNC(keyUp);

private :
    Scene3d* scene;
};

#endif GAME_H_
