
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
        L"window_launcher",
        1000,
        650,
        1,
        60,
        false,
        CS_HREDRAW | CS_VREDRAW,
        IDI_APPLICATION,
        IDC_ARROW,
        WHITE_BRUSH,
        IDI_APPLICATION
    );

    glpa.showWindow(WINDOW_LAUNCHER);

    // glpa.updateWindowInfo(WINDOW_LAUNCHER);


    // // Launcher Class Registration
    // WNDCLASSEX windowClassEx_LAU = WndMain.registerClass
    // (
    //     CS_HREDRAW | CS_VREDRAW,
    //     WndLAU.wndProc,
    //     0,
    //     0,
    //     hInstance,
    //     IDI_APPLICATION,
    //     IDC_ARROW,
    //     WHITE_BRUSH,
    //     NULL,
    //     L"window_LAU",
    //     IDI_APPLICATION
    // );

    // if (!WndMain.checkClass(&windowClassEx_LAU))
    // {
    //     return 1;
    // }

    // // Play Class Registration
    // WNDCLASSEX windowClassEx_PLAY = WndMain.registerClass
    // (
    //     CS_HREDRAW | CS_VREDRAW,
    //     WndPLAY.wndProc,
    //     0,
    //     0,
    //     hInstance,
    //     IDI_APPLICATION,
    //     IDC_ARROW,
    //     WHITE_BRUSH,
    //     NULL,
    //     L"window_PLAY",
    //     IDI_APPLICATION
    // );

    // if (!WndMain.checkClass(&windowClassEx_PLAY))
    // {
    //     return 1;
    // }

    // // Creation of WndLAU window
    // WndLAU.hWnd = CreateWindow(             // HWND window handle
    //     L"window_LAU",                      // LPCSTR Registered class name address
    //     L"LAUNCHER",                        // LPCSTR Window text address
    //     WS_OVERLAPPEDWINDOW,                // DWORD Window style, which can be specified with the parameter WS_MESSAGENAME
    //     CW_USEDEFAULT, CW_USEDEFAULT,       // int Window horizontal coordinate position, Window vertical coordinate position
    //     WndLAU.windowSize.width, WndLAU.windowSize.height,  // int Window Width, Window Height
    //     HWND_DESKTOP,                       // HWND Parent Window Handle
    //     NULL,                               // HMENU Menu handle or child window ID
    //     hInstance,                          // HINSTANCE Application instance handle
    //     NULL                                // void FAR* Address of window creation data
    // );

    // if (!WndMain.checkWindow(WndLAU.hWnd))
    // {
    //     return 1;
    // }

    // // Storing WinMain Function Arguments
    // WndMain.hInstance = hInstance;
    // WndMain.nCmdShow = nCmdShow;

    // ShowWindow(
    //     WndLAU.hWnd,
    //     nCmdShow
    // );

    MSG msg;        //メッセージ構造体

    while (true) {
    // Returns 1 (true) if a message is retrieved and 0 (false) if not.
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            // Exit from the loop when the exit message comes.
            break;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    } 
        // else if (WndPLAY.state.focus)
        // {
        //     WndPLAY.fpsSystem.fpsLimiter();

        //     PatBlt(
        //         WndPLAY.buffer.hBufDC, 
        //         0, 
        //         0, 
        //         WINDOW_WIDTH * DISPLAY_RESOLUTION, 
        //         WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
        //         WHITENESS
        //     );
        //     scrPLAYDwgContModif(WndPLAY.buffer.hBufDC);

        //     InvalidateRect(WndPLAY.hWnd, NULL, FALSE);
        // }
        // else if (WndLAU.state.focus)
        // {
        //     WndLAU.fpsSystem.fpsLimiter();

        //     PatBlt(
        //         WndLAU.buffer.hBufDC, 
        //         0, 
        //         0, 
        //         WINDOW_WIDTH * DISPLAY_RESOLUTION, 
        //         WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
        //         WHITENESS
        //     );
        //     scrLAUDwgContModif(WndLAU.buffer.hBufDC);

        //     InvalidateRect(WndLAU.hWnd, NULL, FALSE);
        // }
        
    }

    // When the function receives a WM_QUIT message and exits, the wParam parameter of the message is
    // Returns the exit code that has If the function exits before entering the message loop, return 0
    return (int)msg.wParam;             
}
