#include <windows.h>

int WINAPI WinMain
(
	_In_ HINSTANCE hInstance,			//�A�v���P�[�V�����̃C���X�^���X�n���h��
	_In_ HINSTANCE hPrevInstance,		//�A�v���P�[�V�����ȑO�̃C���X�^���X�n���h��������BWin32�A�v���P�[�V�����ł͏��NULL
	_In_ LPSTR lpCmdLine,				//�R�}���h���C�����i�[���ꂽ�ANULL�ŏI��镶����ւ̃|�C���^������B�v���O�������͊܂܂�Ȃ�
	_In_ int nCmdShow					//�E�B���h�E���ǂ̂悤�ɕ\�����邩�̎w�肪����BSW_MESSAGENAME�̒l������
)
{
	MSG msg;			//���b�Z�[�W�\����
	HWND hwnd;			//�E�B���h�E�n���h��

	hwnd = CreateWindow
	(
		L"STATIC",								//LPCSTR �o�^���ꂽ�N���X���̃A�h���X
		L"�X�^�e�B�b�N�R���g���[��",				//LPCSTR �E�B���h�E�e�L�X�g�̃A�h���X
		SS_CENTER | SS_NOTIFY | WS_VISIBLE,		//DWORD �E�B���h�E�X�^�C���BWS_MESSAGENAME�̃p�����[�^�Ŏw��ł���
		100,									//int �E�B���h�E�̐������W�̈ʒu
		100,									//int �E�B���h�E�̐������W�̈ʒu
		100,									//int �E�B���h�E�̕�
		100,									//int �E�B���h�E�̍���
		HWND_DESKTOP,							//HWND �e�E�B���h�E�̃n���h��
		NULL,									//HMENU ���j���[�̃n���h���܂��͎q�E�B���h�E��ID
		hInstance,								//HINSTANCE �A�v���P�[�V�����C���X�^���X�̃n���h��
		NULL									//void FAR* �E�B���h�E�쐬�f�[�^�̃A�h���X
	);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (msg.message == WM_LBUTTONUP)
		{
			break;
		}
		DispatchMessage(&msg);
	}

	return (int)msg.wParam;			/*�֐���WM_QUIT���b�Z�[�W���󂯎���ďI�������Ƃ��́A���b�Z�[�W��wParam�p�����[�^�����I���R�[�h��Ԃ��B
								�@�֐������b�Z�[�W���[�v�ɓ���O�ɏI�������Ƃ��́A�O��Ԃ�*/
}