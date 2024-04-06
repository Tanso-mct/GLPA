
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
)                      
{
    glpa.initialize(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    glpa.createWindow(
        WINDOW_CONSOLE, WINDOWCLASS_CONSOLE, 1000, 650, 1, 60, CS_HREDRAW | CS_VREDRAW,
        IDI_APPLICATION, IDC_ARROW, WHITE_BRUSH, IDI_APPLICATION, false, true, WS_SYSMENU
    );

    glpa.updateWindow(WINDOW_CONSOLE, GLPA_WINDOW_STATUS_DEF);
    // glpa.setSingleWindow(true);

    glpa.createScene(SCENE_GLPA_CONSOLE, GLPA_SCENE_2D);
    glpa.loadScene(SCENE_GLPA_CONSOLE, SCENE_FOLDER_PASS_GLPA_CONSOLE);
    
    glpa.selectUseScene(WINDOW_CONSOLE, SCENE_GLPA_CONSOLE);

    // Not Glpa
    Console console(glpa.getPtScene2d(SCENE_GLPA_CONSOLE));

    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE, L"console_keyDown", 
        GLPA_USER_FUNC_PT(console, keyDown), GLPA_USERINPUT_MESSAGE_KEYDOWN
    );
    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE, L"console_keyUp", 
        GLPA_USER_FUNC_PT(console, keyUp), GLPA_USERINPUT_MESSAGE_KEYUP
    );
    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE, L"console_mouseLbtnDown", 
        GLPA_USER_FUNC_PT(console, mouseLbtnDown), GLPA_USERINPUT_MESSAGE_MOUSELBTNDOWN
    );
    glpa.setSceneFrameFunc(SCENE_GLPA_CONSOLE, L"console_main_update", GLPA_SCENE_FUNC_PT(console, mainUpdate));


    Game game;

    console.command.add(L"start", COMMAND_FUN_PT(game, tempStart));
    console.command.add(L"release", COMMAND_FUN_PT(game, tempRelease));
    // console.command.add(L"reset", COMMAND_FUN_PT(game, camReset));

    glpa.runGraphicLoop();

    glpa.releaseUserInputFunc(L"console_keyDown");
    glpa.releaseUserInputFunc(L"console_keyUp");
    glpa.releaseUserInputFunc(L"console_mouseLbtnDown");
    glpa.releaseSceneFrameFunc(SCENE_GLPA_CONSOLE, L"console_main_update");

    console.command.release(L"start");
    console.command.release(L"release");
    // console.command.release(L"reset");


    glpa.releaseScene(SCENE_GLPA_CONSOLE);

    return (int)glpa.msg.wParam;             
}
