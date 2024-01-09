#ifndef GAME_H_
#define GAME_H_

#include "glpa.h"

#include "command.h"


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

private :
};

#endif GAME_H_
