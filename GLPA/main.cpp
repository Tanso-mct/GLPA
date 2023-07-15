
#include "main.h"

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,          //�A�v���P�[�V�����̃C���X�^���X�n���h��
    _In_opt_ HINSTANCE hPrevInstance,  //�A�v���P�[�V�����ȑO�̃C���X�^���X�n���h��������BWin32�A�v���P�[�V�����ł͏��NULL
    _In_ LPSTR lpCmdLine,              //�R�}���h���C�����i�[���ꂽ�ANULL�ŏI��镶����ւ̃|�C���^������B
                                       //�v���O�������͊܂܂�Ȃ�
    _In_ int nCmdShow                  //�E�B���h�E���ǂ̂悤�ɕ\�����邩�̎w�肪����BSW_MESSAGENAME�̒l������
)                      
{
    WNDCLASSEX wcex_LAU;

    wcex_LAU.cbSize = sizeof(wcex_LAU);                            //UINT WNDCLASSEX�\���̂̑傫���̐ݒ�
    wcex_LAU.style = CS_HREDRAW | CS_VREDRAW;                      //UINT �N���X�X�^�C����\���BCS_MESSAGENAME�̒l��O
                                                                   //R���Z�q�őg�ݍ��킹���l�ƂȂ�
    wcex_LAU.lpfnWndProc = WndLAU.wndProc;                         //WNDPROC WNDPROC���w���|�C���^
    wcex_LAU.cbClsExtra = 0;                                       //int �E�B���h�E�N���X�\���̂̐ՂɊ��蓖�Ă�o�C�g��������
    wcex_LAU.cbWndExtra = 0;                                       //int �E�B���h�E�C���X�^���X�̐ՂɊ��蓖�Ă�o�C�g��������
    wcex_LAU.hInstance = hInstance;                                //HINSTANCE �C���X�^���X�n���h��
    wcex_LAU.hIcon =                                               //HICON �N���X�A�C�R�����w�肷��
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_LAU.hCursor =                                             //HCURSOR �N���X�J�[�\�����w�肷��
        LoadCursor(NULL, IDC_ARROW);
    wcex_LAU.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);           //HBRUSH �N���X�w�i�u���V���w�肷��
    wcex_LAU.lpszMenuName = NULL;                                  //LPCSTR �N���X���j���[�̃��\�[�X�����w�肷��
    wcex_LAU.lpszClassName = L"window_LAU";                        //LPCSTR �E�B���h�E�N���X�̖��O���w�肷��
    wcex_LAU.hIconSm =                                             //HICON �����ȃN���X�A�C�R�����w�肷��
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

    wcex_PLAY.cbSize = sizeof(wcex_PLAY);                                //UINT WNDCLASSEX�\���̂̑傫���̐ݒ�
    wcex_PLAY.style = CS_HREDRAW | CS_VREDRAW;                           //UINT �N���X�X�^�C����\���BCS_MESSAGENAME�̒l��OR���Z�q�őg�ݍ��킹���l�ƂȂ�
    wcex_PLAY.lpfnWndProc = WndPLAY.wndProc;                             //WNDPROC WNDPROC���w���|�C���^
    wcex_PLAY.cbClsExtra = 0;                                            //int �E�B���h�E�N���X�\���̂̐ՂɊ��蓖�Ă�o�C�g��������
    wcex_PLAY.cbWndExtra = 0;                                            //int �E�B���h�E�C���X�^���X�̐ՂɊ��蓖�Ă�o�C�g��������
    wcex_PLAY.hInstance = hInstance;                                     //HINSTANCE �C���X�^���X�n���h��
    wcex_PLAY.hIcon =                                                    //HICON �N���X�A�C�R�����w�肷��
        LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex_PLAY.hCursor =                                                  //HCURSOR �N���X�J�[�\�����w�肷��
        LoadCursor(NULL, IDC_ARROW);
    wcex_PLAY.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);                //HBRUSH �N���X�w�i�u���V���w�肷��
    wcex_PLAY.lpszMenuName = NULL;                                       //LPCSTR �N���X���j���[�̃��\�[�X�����w�肷��
    wcex_PLAY.lpszClassName = L"window_PLAY";                            //LPCSTR �E�B���h�E�N���X�̖��O���w�肷��
    wcex_PLAY.hIconSm =                                                  //HICON �����ȃN���X�A�C�R�����w�肷��
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

    WndLAU.hWnd = CreateWindow(             //HWND �E�B���h�E�n���h��
        L"window_LAU",                      //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
        L"LAUNCHER",                        //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
        WS_OVERLAPPEDWINDOW,                //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
        CW_USEDEFAULT, CW_USEDEFAULT,       //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
        WndLAU.wndWidth, WndLAU.wndHeight,  //int �E�B���h�E�̕�, �E�B���h�E�̍���
        HWND_DESKTOP,                       //HWND �e�E�B���h�E�̃n���h��
        NULL,                               //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
        hInstance,                          //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
        NULL                                //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
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

    MSG msg;        //���b�Z�[�W�\����

    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return (int)msg.wParam;             //�֐���WM_QUIT���b�Z�[�W���󂯎���ďI�������Ƃ��́A���b�Z�[�W��wParam�p�����[�^��
							            //���I���R�[�h��Ԃ��B�֐������b�Z�[�W���[�v�ɓ���O�ɏI�������Ƃ��́A�O��Ԃ�
}
