
#include "userinput.h"

#include "windowbefore.h"

// WND LAU UserInput define

void WndLAUInput ::keyDown(WPARAM wParam)
{
        switch (wParam)
        {
        case VK_ESCAPE :
                _stprintf_s(szstr, _T("%s"), _T("ESCAPE"));
                // OutputDebugStringW(_T("ESCAPE\n"));
                break;
        case VK_SPACE :
                _stprintf_s(szstr, _T("%s"), _T("SPACE"));
                WndPLAY.hWnd = CreateWindow(           //HWND �E�B���h�E�n���h��
                    L"window_PLAY",                 //LPCSTR �o�^���ꂽ�N���X���̃A�h���X
                    L"PLAY",                        //LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
                    WS_OVERLAPPEDWINDOW,            //DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
                    CW_USEDEFAULT, CW_USEDEFAULT,   //int �E�B���h�E�̐������W�̈ʒu, �E�B���h�E�̐������W�̈ʒu
                    WndPLAY.windowSize.width, WndPLAY.windowSize.height,    //int �E�B���h�E�̕�, �E�B���h�E�̍���
                    HWND_DESKTOP,                   //HWND �e�E�B���h�E�̃n���h��
                    NULL,                           //HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
                    WndMain.hInstance,           //HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
                    NULL                            //void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
                );

                if (!WndPLAY.hWnd)
                {
                        MessageBox(
                        NULL,
                        _T("window_PLAY make fail"),
                        _T("window_PLAY"),
                        MB_ICONEXCLAMATION
                        );

                    // return 1;
                }

                ShowWindow(
                        WndPLAY.hWnd,
                        WndMain.nCmdShow
                );

                UpdateWindow(WndPLAY.hWnd);
                // OutputDebugStringW(_T("SPACE\n"));
                break;
        case VK_SHIFT :
                _stprintf_s(szstr, _T("%s"), _T("SHIFT"));
                // OutputDebugStringW(_T("SHIFT\n"));
                break;
        case 'W' :
                // _stprintf_s(szstr, _T("%s"), _T("W ON"));
                // WndLAU.fpsSystem.setFps = 240;
                // deve001.cam.wPos.y -= 10;
                // _stprintf_s(szstr, _T("%lf"), (deve001.cam.wPos.y));
                // OutputDebugStringW(_T("W\n"));
                break;
        case 'S' :
                // _stprintf_s(szstr, _T("%s"), _T("S ON"));
                // deve001.cam.wPos.y += 10;
                // _stprintf_s(szstr, _T("%lf"), (deve001.cam.wPos.y));
                // WndLAU.fpsSystem.setFps = 10;
                break;
        case 'C' :
                // sampleBmpFile.loadBinary(FILETYPE_BMP, ("temp.bmp"));
                // sampleBmpFile.checkBinary();
                
                // if (!deve001.cam.initialized)
                // {
                //         OutputDebugStringW(_T("Camera Initialize\n"));
                //         tempMtlFile.loadData("temp_cube");
                        
                //         // tempObject.inputData("temp_cube");
                //         // tempObject.inputData("temp_secondCube");
                //         tempObject.inputData("temp_third_cube");


                //         // tempObject.inputData("temp_farCube");
                //         deve001.cam.initialize();
                //         // deve001.cam.defViewVolume();
                // }
                OutputDebugStringW(_T("New scene\n"));
                // deve001.cam.coordinateTransRange(&tempObject.data);
                // deve001.cam.clippingRange(tempObject.data);
                // deve001.cam.polyBilateralJudge(tempObject.data);
                // deve001.cam.coordinateTrans(tempObject.data);
                // deve001.cam.polyInViewVolumeJudge(tempObject.data);

                break;
        case 'I' :
                OutputDebugStringW(_T("Camera Initialize\n"));
                tempMtlFile.loadData("temp_cube");
                tempObject.inputData("temp_cube");
                tempObject.inputData("temp_farCube");
                // deve001.cam.initialize();
                // deve001.cam.defViewVolume();
                break;
        default :
                _stprintf_s(szstr, _T("%s"), _T("ANY"));
                break;
        }
};

void WndLAUInput ::keyUp(WPARAM wParam)
{
        switch (wParam)
        {
                case VK_ESCAPE :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("ESCAPE UP\n"));
                        break;
                case VK_SPACE :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SPACE UP\n"));
                        break;
                case VK_SHIFT :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SHIFT UP\n"));
                        break;
                case 'W' :
                        // _stprintf_s(szstr, _T("%s"), _T("W OFF"));
                        // OutputDebugStringW(_T("W UP\n"));
                        break;
                case 'S' :
                        // _stprintf_s(szstr, _T("%s"), _T("S OFF"));
                        break;
                default :
                        _stprintf_s(szstr, _T("%s"), _T("NAN"));
                        break;
        }
};

void WndLAUInput :: mouseMove(LPARAM lParam)
{
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};


void WndLAUInput :: mouseLbtnDown(LPARAM lParam)
{
        pt.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
        pt.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};

// WND Play UserInput define
void WndPLAYInput ::keyDown(WPARAM wParam)
{
        switch (wParam)
        {
        case VK_ESCAPE :
                _stprintf_s(szstrfPlay, _T("%s"), _T("ESCAPE"));
                // OutputDebugStringW(_T("ESCAPE\n"));
                break;
        case VK_SPACE :
                _stprintf_s(szstrfPlay, _T("%s"), _T("SPACE"));
        
                break;
        case VK_SHIFT :
                _stprintf_s(szstrfPlay, _T("%s"), _T("SHIFT"));
                // OutputDebugStringW(_T("SHIFT\n"));
                break;
        case 'W' :
                _stprintf_s(szstrfPlay, _T("%s"), _T("W ON"));
                // OutputDebugStringW(_T("W\n"));
                break;
        default :
                _stprintf_s(szstrfPlay, _T("%s"), _T("ANY"));
                break;
        }
};

void WndPLAYInput ::keyUp(WPARAM wParam)
{
        switch (wParam)
        {
                case VK_ESCAPE :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("ESCAPE UP\n"));
                        break;
                case VK_SPACE :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SPACE UP\n"));
                        break;
                case VK_SHIFT :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        // OutputDebugStringW(_T("SHIFT UP\n"));
                        break;
                case 'W' :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("W OFF"));
                        // OutputDebugStringW(_T("W UP\n"));
                        break;
                default :
                        _stprintf_s(szstrfPlay, _T("%s"), _T("NAN"));
                        break;
        }
};

void WndPLAYInput :: mouseMove(LPARAM lParam)
{
        ptfPlay.x = LOWORD(lParam) * DISPLAY_RESOLUTION;
        ptfPlay.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        // _stprintf_s(szstr, _T("%d,%d"), pt.x, pt.y);
};


void WndPLAYInput :: mouseLbtnDown(LPARAM lParam)
{
        ptfPlay.x = LOWORD(lParam) * DISPLAY_RESOLUTION;  
        ptfPlay.y = HIWORD(lParam) * DISPLAY_RESOLUTION;
        _stprintf_s(szstrfPlay, _T("%d,%d"), ptfPlay.x, ptfPlay.y);
};


WndLAUInput UserInputWndLAU;
WndPLAYInput UserInputWndPLAY;
