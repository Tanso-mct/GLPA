#include <windows.h>

int WINAPI WinMain
(
	_In_ HINSTANCE hInstance,			//アプリケーションのインスタンスハンドル
	_In_ HINSTANCE hPrevInstance,		//アプリケーション以前のインスタンスハンドルが入る。Win32アプリケーションでは常にNULL
	_In_ LPSTR lpCmdLine,				//コマンドラインが格納された、NULLで終わる文字列へのポインタが入る。プログラム名は含まれない
	_In_ int nCmdShow					//ウィンドウをどのように表示するかの指定が入る。SW_MESSAGENAMEの値が入る
)
{
	MSG msg;			//メッセージ構造体
	HWND hwnd;			//ウィンドウハンドル

	hwnd = CreateWindow
	(
		L"STATIC",								//LPCSTR 登録されたクラス名のアドレス
		L"スタティックコントロール",				//LPCSTR ウィンドウテキストのアドレス
		SS_CENTER | SS_NOTIFY | WS_VISIBLE,		//DWORD ウィンドウスタイル。WS_MESSAGENAMEのパラメータで指定できる
		100,									//int ウィンドウの水平座標の位置
		100,									//int ウィンドウの垂直座標の位置
		100,									//int ウィンドウの幅
		100,									//int ウィンドウの高さ
		HWND_DESKTOP,							//HWND 親ウィンドウのハンドル
		NULL,									//HMENU メニューのハンドルまたは子ウィンドウのID
		hInstance,								//HINSTANCE アプリケーションインスタンスのハンドル
		NULL									//void FAR* ウィンドウ作成データのアドレス
	);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (msg.message == WM_LBUTTONUP)
		{
			break;
		}
		DispatchMessage(&msg);
	}

	return (int)msg.wParam;			/*関数がWM_QUITメッセージを受け取って終了したときは、メッセージのwParamパラメータが持つ終了コードを返す。
								　関数がメッセージループに入る前に終了したときは、０を返す*/
}