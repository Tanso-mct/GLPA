//--------------------------------//
//
//内容                                                 
//1:UserInput定義
//2:WND LAU UserInputの定義
//3:WND PLAY UserInputの定義
//
//--------------------------------//

//1:UserInput定義


//2:WND LAU UserInputの定義

void WndLAUInput : keyDown(WPARAM w_Param)
{
    switch (w_Param)
    {
        case VK_ESCAPE :
                _stprintf_s(szstr, _T("%s"), _T("ESCAPE"));
                // OutputDebugStringW(_T("ESCAPE\n"));
                break;
        case VK_SPACE :
                _stprintf_s(szstr, _T("%s"), _T("SPACE"));
                hWnd1Open = false;
                hWnd_PLAY = CreateWindow(           //HWND ウィンドウハンドル
                    L"window_PLAY",                 //LPCSTR 登録されたクラス名のアドレス
                    L"PLAY",                       //LPCSTR ウィンドウテキストのアドレス
                    WS_OVERLAPPEDWINDOW,            //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
                    CW_USEDEFAULT, CW_USEDEFAULT,   //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
                    WINDOW_WIDTH, WINDOW_HEIGHT,    //int ウィンドウの幅, ウィンドウの高さ
                    HWND_DESKTOP,                   //HWND 親ウィンドウのハンドル
                    NULL,                           //HMENU メニューのハンドルまたは子ウィンドウのID
                    gr_hInstance,                   //HINSTANCE アプリケーションインスタンスのハンドル
                    NULL                            //void FAR* ウィンドウ作成データのアドレス
                );

                if (!hWnd_PLAY)
                {
                    MessageBox(
                        NULL,
                        _T("window_PLAY make fail"),
                        _T("window_PLAY"),
                        MB_ICONEXCLAMATION
                    );

                    return 1;
                }

                ShowWindow(
                    hWnd_PLAY,
                    gr_nCmdShow
                );

                UpdateWindow(hWnd_PLAY);
                // OutputDebugStringW(_T("SPACE\n"));
                break;
        case VK_SHIFT :
                _stprintf_s(szstr, _T("%s"), _T("SHIFT"));
                // OutputDebugStringW(_T("SHIFT\n"));
                break;
        case 'W' :
                _stprintf_s(szstr, _T("%s"), _T("W ON"));
                // OutputDebugStringW(_T("W\n"));
                break;
        default :
                _stprintf_s(szstr, _T("%s"), _T("ANY"));
                break;
    }
};

void WndLAUInput : keyUp(WPARAM w_Param)
{
    
};

void WndLAUInput : mouseLbtnDown(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseLbtnUp(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseLbtnDblclick(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseRbtnDown(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseRbtnUp(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseRbtnDblClick(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseMbtnDown(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseMbtnUp(LPARAM l_Param)
{
    
};

void WndLAUInput : mouseMbtnWheel(LPARAM l_Param)
{
    
};

//3:WND PLAY UserInputの定義

void WndPLAYInput : keyDown(WPARAM w_Param)
{

};

void WndPLAYInput : keyUp(WPARAM w_Param)
{
    
};

void WndPLAYInput : mouseLbtnDown(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseLbtnUp(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseLbtnDblclick(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseRbtnDown(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseRbtnUp(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseRbtnDblClick(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseMbtnDown(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseMbtnUp(LPARAM l_Param)
{
    
};

void WndPLAYInput : mouseMbtnWheel(LPARAM l_Param)
{
    
};

