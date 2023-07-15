
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          //アプリケーションのインスタンスハンドル
    _In_opt_ HINSTANCE hPrevInstance,  //アプリケーション以前のインスタンスハンドルが入る。Win32アプリケーションでは常にNULL
    _In_ LPSTR lpCmdLine,              //コマンドラインが格納された、NULLで終わる文字列へのポインタが入る。
                                       //プログラム名は含まれない
    _In_ int nCmdShow                  //ウィンドウをどのように表示するかの指定が入る。SW_MESSAGENAMEの値が入る
)                      
{
    WNDCLASSEX wcex_LAU;

    wcex_LAU.cbSize = sizeof(wcex_LAU);                            //UINT WNDCLASSEX構造体の大きさの設定
    wcex_LAU.style = CS_HREDRAW | CS_VREDRAW;                      //UINT クラススタイルを表す。CS_MESSAGENAMEの値をO
                                                                   //R演算子で組み合わせた値となる
    wcex_LAU.lpfnWndProc = WndLAU.wndProc;                         //WNDPROC WNDPROCを指すポインタ
    wcex_LAU.cbClsExtra = 0;                                       //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex_LAU.cbWndExtra = 0;                                       //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex_LAU.hInstance = hInstance;                                //HINSTANCE インスタンスハンドル
    wcex_LAU.hIcon =                                               //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_LAU.hCursor =                                             //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex_LAU.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);           //HBRUSH クラス背景ブラシを指定する
    wcex_LAU.lpszMenuName = NULL;                                  //LPCSTR クラスメニューのリソース名を指定する
    wcex_LAU.lpszClassName = L"window_LAU";                        //LPCSTR ウィンドウクラスの名前を指定する
    wcex_LAU.hIconSm =                                             //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex_LAU.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex_LAU))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WNDCLASSEX wcex_PLAY;        //struct tagWNDCLASSEXW

    wcex_PLAY.cbSize = sizeof(wcex_PLAY);                                //UINT WNDCLASSEX構造体の大きさの設定
    wcex_PLAY.style = CS_HREDRAW | CS_VREDRAW;                           //UINT クラススタイルを表す。CS_MESSAGENAMEの値をOR演算子で組み合わせた値となる
    wcex_PLAY.lpfnWndProc = WndPLAY.wndProc;                             //WNDPROC WNDPROCを指すポインタ
    wcex_PLAY.cbClsExtra = 0;                                            //int ウィンドウクラス構造体の跡に割り当てるバイト数を示す
    wcex_PLAY.cbWndExtra = 0;                                            //int ウィンドウインスタンスの跡に割り当てるバイト数を示す
    wcex_PLAY.hInstance = hInstance;                                     //HINSTANCE インスタンスハンドル
    wcex_PLAY.hIcon =                                                    //HICON クラスアイコンを指定する
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_PLAY.hCursor =                                                  //HCURSOR クラスカーソルを指定する
        LoadCursor(NULL, IDC_ARROW);
    wcex_PLAY.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH クラス背景ブラシを指定する
    wcex_PLAY.lpszMenuName = NULL;                                       //LPCSTR クラスメニューのリソース名を指定する
    wcex_PLAY.lpszClassName = L"window_PLAY";                            //LPCSTR ウィンドウクラスの名前を指定する
    wcex_PLAY.hIconSm =                                                  //HICON 小さなクラスアイコンを指定する
        LoadIcon(wcex_PLAY.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    if (!RegisterClassEx(&wcex_PLAY))
    {
        MessageBox(
            NULL,
            _T("RegisterClassEx fail"),
            _T("window_PLAY"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WndLAU.hWnd = CreateWindow(             //HWND ウィンドウハンドル
        L"window_LAU",                      //LPCSTR 登録されたクラス名のアドレス
        L"LAUNCHER",                        //LPCSTR ウィンドウテキストのアドレス
        WS_OVERLAPPEDWINDOW,                //DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
        CW_USEDEFAULT, CW_USEDEFAULT,       //int ウィンドウの水平座標の位置, ウィンドウの垂直座標の位置
        WndLAU.wndWidth, WndLAU.wndHeight,  //int ウィンドウの幅, ウィンドウの高さ
        HWND_DESKTOP,                       //HWND 親ウィンドウのハンドル
        NULL,                               //HMENU メニューのハンドルまたは子ウィンドウのID
        hInstance,                          //HINSTANCE アプリケーションインスタンスのハンドル
        NULL                                //void FAR* ウィンドウ作成データのアドレス
    );

    if (!WndLAU.hWnd)
    {
        MessageBox(
            NULL,
            _T("window make fail"),
            _T("window_LAU"),
            MB_ICONEXCLAMATION
        );

        return 1;
    }

    WndMain.gr_hInstance = hInstance;
    WndMain.gr_nCmdShow = nCmdShow;

    ShowWindow(
        WndLAU.hWnd,
        nCmdShow
    );
    UpdateWindow(WndLAU.hWnd);

    MSG msg;        //メッセージ構造体

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return (int)msg.wParam;             //関数がWM_QUITメッセージを受け取って終了したときは、メッセージのwParamパラメータが
							            //持つ終了コードを返す。関数がメッセージループに入る前に終了したときは、０を返す
}
