//--------------------------------//
//
//���e                                                 
//1:UserInput��`
//2:WND LAU UserInput�̒�`
//3:WND PLAY UserInput�̒�`
//
//--------------------------------//

//1:UserInput��`


//2:WND LAU UserInput�̒�`

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
                hWnd_PLAY = CreateWindow(           //HWND �E�B���h�E�n���h��
                    L"window_PLAY",                 //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
                    L"PLAY",                       //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
                    WS_OVERLAPPEDWINDOW,            //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
                    CW_USEDEFAULT, CW_USEDEFAULT,   //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
                    WINDOW_WIDTH, WINDOW_HEIGHT,    //int �E�B���h�E�̕�, �E�B���h�E�̍���
                    HWND_DESKTOP,                   //HWND �e�E�B���h�E�̃n���h��
                    NULL,                           //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
                    gr_hInstance,                   //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
                    NULL                            //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
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

//3:WND PLAY UserInput�̒�`

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

