
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          // Application instance handle
    _In_opt_ HINSTANCE hPrevInstance,  // Contains the instance handle before the application; always NULL for Win32 applications
    _In_ LPSTR lpCmdLine,              // Contains a pointer to a null-terminated string containing the command line.
                                       // Program name not included
    _In_ int nCmdShow                  // Contains the value of SW_MESSAGENAME.
)                      
{
    glpa.initialize(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    glpa.createWindow
    (
        WINDOW_CONSOLE,
        WINDOWCLASS_CONSOLE,
        1000,
        650,
        1,
        60,
        CS_HREDRAW | CS_VREDRAW,
        IDI_APPLICATION,
        IDC_ARROW,
        WHITE_BRUSH,
        IDI_APPLICATION,
        false,
        true
    );

    glpa.updateWindow(WINDOW_CONSOLE, GLPA_WINDOW_STATUS_DEF);
    glpa.setSingleWindow(true);

    glpa.createScene(SCENE_GLPA_CONSOLE, GLPA_SCENE_2D);
    glpa.loadScene(SCENE_GLPA_CONSOLE, SCENE_FOLDER_PASS_GLPA_CONSOLE);
    
    glpa.selectUseScene(WINDOW_CONSOLE, SCENE_GLPA_CONSOLE);

    Console temp;

    temp.setScenePt(glpa.getPtScene2d(SCENE_GLPA_CONSOLE));

    glpa.setUserInputFunc(
        SCENE_GLPA_CONSOLE,
        L"tempTyping",
        [&temp](std::string scene_name, WPARAM wParam, LPARAM lParam) {
            temp.tempTyping(scene_name, wParam, lParam);
        },
        GLPA_USERINPUT_MESSAGE_KEYDOWN
    );

    glpa.runGraphicLoop();

    glpa.releaseScene(SCENE_GLPA_CONSOLE);

    // When the function receives a WM_QUIT message and exits, the wParam parameter of the message is
    // Returns the exit code that has If the function exits before entering the message loop, return 0
    return (int)glpa.msg.wParam;             
}
