
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          //�A�v���P�[�V�����̃C���X�^���X�n���h��
    _In_opt_ HINSTANCE hPrevInstance,  //�A�v���P�[�V�����ȑO�̃C���X�^���X�n���h��������BWin32�A�v���P�[�V�����ł͏��NULL
    _In_ LPSTR lpCmdLine,              //�R�}���h���C�����i�[���ꂽ�ANULL�ŏI��镶����ւ̃|�C���^������B
                                       //�v���O�������͊܂܂�Ȃ�
    _In_ int nCmdShow                  //�E�B���h�E���ǂ̂悤�ɕ\�����邩�̎w�肪����BSW_MESSAGENAME�̒l������
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
    WndLAU.hWnd = CreateWindow(             //HWND �E�B���h�E�n���h��
        L"window_LAU",                      //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
        L"LAUNCHER",                        //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
        WS_OVERLAPPEDWINDOW,                //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
        CW_USEDEFAULT, CW_USEDEFAULT,       //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
        WndLAU.windowSize.width, WndLAU.windowSize.height,  //int �E�B���h�E�̕�, �E�B���h�E�̍���
        HWND_DESKTOP,                       //HWND �e�E�B���h�E�̃n���h��
        NULL,                               //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
        hInstance,                          //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
        NULL                                //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
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

    MSG msg;        //���b�Z�[�W�\����

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

    
    
    return (int)msg.wParam;             //�֐���WM_QUIT���b�Z�[�W���󂯎���ďI�������Ƃ��́A���b�Z�[�W��wParam�p�����[�^��
							            //���I���R�[�h��Ԃ��B�֐������b�Z�[�W���[�v�ɓ���O�ɏI�������Ƃ��́A�O��Ԃ�
}
