
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine, _In_ int nCmdShow
)                      
{
    glpa.initialize(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    glpa.createWindow
    (
        WINDOW_CONSOLE, WINDOWCLASS_CONSOLE, 1000, 650, 1, 60, CS_HREDRAW | CS_VREDRAW,
        IDI_APPLICATION, IDC_ARROW, WHITE_BRUSH, IDI_APPLICATION, false, true
    );

    glpa.updateWindow(WINDOW_CONSOLE, GLPA_WINDOW_STATUS_DEF);
    glpa.setSingleWindow(true);

    glpa.createScene(SCENE_GLPA_CONSOLE, GLPA_SCENE_2D);
    glpa.loadScene(SCENE_GLPA_CONSOLE, SCENE_FOLDER_PASS_GLPA_CONSOLE);
    
    glpa.selectUseScene(WINDOW_CONSOLE, SCENE_GLPA_CONSOLE);

    // Not Glpa
    Console temp(&glpa);
    temp.setScenePt(glpa.getPtScene2d(SCENE_GLPA_CONSOLE));

    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE, L"tempTypingDown", GLPA_USER_FUNC_PT(temp, tempTypingDown), GLPA_USERINPUT_MESSAGE_KEYDOWN
    );

    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE, L"tempTypingUp", GLPA_USER_FUNC_PT(temp, tempTypingUp), GLPA_USERINPUT_MESSAGE_KEYUP
    );

    glpa.runGraphicLoop();

    glpa.releaseUserInputFunc(L"tempTypingDown");
    glpa.releaseUserInputFunc(L"tempTypingUp");

    glpa.releaseScene(SCENE_GLPA_CONSOLE);

    return (int)glpa.msg.wParam;             
}
