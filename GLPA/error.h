/*
GLPA�G���[������

Visual studio�ł̊J���̏ꍇ�A�G���[�ӏ��̃}�N����Ctrl + �N���b�N�ł��̃t�@�C���ֈړ����܂��B
�e�G���[�̌������L�q���Ă��܂��B�w���ɏ]���C�����s���Ă��������B

�܂��A�J���҂͓��{�l�Ȃ��߂܂����{��ŋL�q�������DeepL�Ŗ|�󂵂Ă��܂��B
�|����e�̊m�F�͊J���Җ{�l���s���܂������A�p�ꂪ���S�ł͂Ȃ����ߌ�󂪑��݂���\��������܂��B���ӂ��Ă��������B

GLPA Error Instructions

For development in Visual studio, Ctrl + click on the macro in the error location to move to this file. 
The causes of each error are described. Please follow the instructions to correct the errors.

The developer is Japanese, so the text is first written in Japanese, and then translated by DeepL.
The translations have been checked by the developer himself, however, 
the English is not perfect and mistranslations may exist. Please be careful.
*/

#ifndef ERROR_H_
#define ERROR_H_

#include <stdexcept>

/*
�����́uLPCWSTR wndName�v�̒l���X�C�b�`���̏����ƈ�v���܂���ł����B
���̏������s���܂���ł����B�����̒l���X�C�b�`���̏����̂����ꂩ�ɂ��Ă��������B
�����́uwindow.h�v�t�@�C����28�s�ڂ́uWINDOW_STATUS�v�Ŏn�܂���̂�ł��B

The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
Conditions are those beginning with "WINDOW_STATUS" at line 28 of file "window.h". 
*/
#define ERROR_GLPA_UPDATE_WINDOW NULL

/*
Png�t�@�C����ǂݍ��߂܂���ł����B�t�@�C���p�X���Ⴄ�Ǝv���܂��B
load�֐��̈����֐������t�@�C���p�X��ݒ肵�Ă��������B

Png file could not be loaded. The file path seems to be different.
Set the correct file path to the Load function argument.
*/
#define ERROR_PNG_LOAD NULL


#endif ERROR_H�ł�