#include "game.h"


void Game::createWnd(){
    glpa.createWindow(
        WINDOW_GAME, WINDOW_CLASS_GAME, 1920, 1080, 1, 60, CS_HREDRAW | CS_VREDRAW,
        IDI_APPLICATION, IDC_ARROW, WHITE_BRUSH, IDI_APPLICATION, false, true, WS_SYSMENU
    );

    glpa.updateWindow(WINDOW_GAME, GLPA_WINDOW_STATUS_DEF);
}


void Game::createScene(){
    glpa.createScene(SCENE_GAME_1, GLPA_SCENE_3D);
}


void Game::loadScene(){
    glpa.loadScene(SCENE_GAME_1, SCENE_FOLDER_PASS_GAME);
}


void Game::selectScene(){
    glpa.selectUseScene(WINDOW_GAME, SCENE_GAME_1);
}


void Game::tempStart(){
    createWnd();
    createScene();
    selectScene();
    loadScene();
}


void Game::tempRelease(){
    glpa.releaseScene(SCENE_GAME_1);
}
