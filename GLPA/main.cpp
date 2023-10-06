
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          //アプリケーションのインスタンスハンドル
    _In_opt_ HINSTANCE hPrevInstance,  //アプリケーション以前のインスタンスハンドルが入る。Win32アプリケーションでは常にNULL
    _In_ LPSTR lpCmdLine,              //コマンドラインが格納された、NULLで終わる文字列へのポインタが入る。
                                       //プログラム名は含まれない
    _In_ int nCmdShow                  //ウィンドウをどのように表示するかの指定が入る。SW_MESSAGENAMEの値が入る
)                      
{
    // Launcher Class Registration
    WNDCLASSEX wcex_LAU = WndMain.registerClass
    (
        CS_HREDRAW | CS_VREDRAW,
        WndLAU.wndProc,
        0,
        0,
        hInstance,
        IDI_APPLICATION,
        IDC_ARROW,
        WHITE_BRUSH,
        NULL,
        L"window_LAU",
        IDI_APPLICATION
    );

    if (!WndMain.checkClass(&wcex_LAU))
    {
        return 1;
    }

    // Play Class Registration
    WNDCLASSEX wcex_PLAY = WndMain.registerClass
    (
        CS_HREDRAW | CS_VREDRAW,
        WndPLAY.wndProc,
        0,
        0,
        hInstance,
        IDI_APPLICATION,
        IDC_ARROW,
        WHITE_BRUSH,
        NULL,
        L"window_PLAY",
        IDI_APPLICATION
    );

    if (!WndMain.checkClass(&wcex_PLAY))
    {
        return 1;
    }

    // Creation of WndLAU window
    WndLAU.hWnd = CreateWindow(             //HWND ウィンドウハンドル
        L"window_LAU",                      //LPCSTR 登録されたクラス名のアドレス
        L"LAUNCHER",                        //LPCSTR ウィンドウテキストのアドレス
        WS_OVERLAPPEDWINDOW,                //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
        CW_USEDEFAULT, CW_USEDEFAULT,       //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
        WndLAU.windowSize.width, WndLAU.windowSize.height,  //int ウィンドウの幅, ウィンドウの高さ
        HWND_DESKTOP,                       //HWND 親ウィンドウのハンドル
        NULL,                               //HMENU メニューのハンドルまたは子ウィンドウのID
        hInstance,                          //HINSTANCE アプリケーションインスタンスのハンドル
        NULL                                //void FAR* ウィンドウ作成データのアドレス
    );

    if (!WndMain.checkWindow(WndLAU.hWnd))
    {
        return 1;
    }

    // Storing WinMain Function Arguments
    WndMain.hInstance = hInstance;
    WndMain.nCmdShow = nCmdShow;

    ShowWindow(
        WndLAU.hWnd,
        nCmdShow
    );

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
        else if (WndPLAY.state.foucus)
        {
            WndPLAY.fpsSystem.fpsLimiter();

            PatBlt(
                WndPLAY.buffer.hBufDC, 
                0, 
                0, 
                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                WHITENESS
            );
            scrPLAYDwgContModif(WndPLAY.buffer.hBufDC);

            InvalidateRect(WndPLAY.hWnd, NULL, FALSE);
        }
        else if (WndLAU.state.foucus)
        {
            WndLAU.fpsSystem.fpsLimiter();

            PatBlt(
                WndLAU.buffer.hBufDC, 
                0, 
                0, 
                WINDOW_WIDTH * DISPLAY_RESOLUTION, 
                WINDOW_HEIGHT * DISPLAY_RESOLUTION, 
                WHITENESS
            );
            scrLAUDwgContModif(WndLAU.buffer.hBufDC);

            InvalidateRect(WndLAU.hWnd, NULL, FALSE);
        }
        
    }

    
    
    return (int)msg.wParam;             //関数がWM_QUITメッセージを受け取って終了したときは、メッセージのwParamパラメータが
							            //持つ終了コードを返す。関数がメッセージループに入る前に終了したときは、０を返す
}
