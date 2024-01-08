#ifndef GAME_H_
#define GAME_H_

#include "glpa.h"

#include "command.h"

class Game{
public :
    Game(Glpa *arg_pt_glpa, Scene3d* arg_pt_scene3d);

    void createWnd();
    void createScene();
    void loadScene();

private :
    Glpa* ptGlpa;
    Scene3d* ptScene3d;
};

#endif GAME_H_
