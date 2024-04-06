#include "game.h"

void Game::createWnd()
{
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

    scene = glpa.getPtScene3d(SCENE_GAME_1);

    keyDownFunc = std::bind(&Game::keyDown, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    glpa.setUserInputFunc(
        SCENE_GAME_1, L"game_key_down", 
        keyDownFunc, GLPA_USERINPUT_MESSAGE_KEYDOWN
    );

    
}


void Game::tempRelease(){
    glpa.releaseUserInputFunc(L"game_key_down");
    glpa.releaseScene(SCENE_GAME_1);
}


// void Game::camReset(){
//     // scene->editCam({0,0,0}, {0,10,0});
// }


void Game::keyDown(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
    switch (wParam){
    case 'Q' :
        scene->rotUseCam({0,10,0});
        break;

    case 'E' :
        scene->rotUseCam({0,-10,0});
        break;

    case 'Z' :
        scene->rotUseCam({10,0,0});
        break;

    case 'C' :
        scene->rotUseCam({-10,0,0});
        break;

    case 'W' :
        scene->moveUseCam({0,0,-50});
        break;

    case 'A' :
        scene->moveUseCam({-50,0,0});
        break;

    case 'S' :
        scene->moveUseCam({0,0,50});
        break;

    case 'D' :
        scene->moveUseCam({50,0,0});
        break;

    case '1' :
        scene->setUseCamTrans({0,0,0}, {0,0,0});

    case VK_SPACE :
        scene->moveUseCam({0,50,0});
        break;

    case VK_SHIFT :
        scene->moveUseCam({0,-50,0});
        break;
    
    default:
        break;
    }

}


// void Game::keyUp(std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){
//     if (selectingTextGroup == GLPA_NULL_WTEXT){
//         return;
//     }

//     std::wstring inputLowWstr = glpa.userInput.convertWParamToLowWstr(wParam);

//     if (scName == SCENE_GLPA_CONSOLE && glpa.userInput.typing){
//         glpa.userInput.typingUpScene2d(scene, selectingTextGroup, wParam, inputLowWstr);
//     }
// }
