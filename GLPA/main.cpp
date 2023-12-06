
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
        WINDOW_LAUNCHER,
        WINDOWCLASS_LAUNCHER,
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

    glpa.updateWindow(WINDOW_LAUNCHER, WINDOW_STATUS_DEF);

    glpa.setSingleWindow(true);

    glpa.runGraphicLoop();

    // When the function receives a WM_QUIT message and exits, the wParam parameter of the message is
    // Returns the exit code that has If the function exits before entering the message loop, return 0
    return (int)glpa.msg.wParam;             
}
