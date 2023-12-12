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
�����́uwindow.h�v�t�@�C���́uGLPA_WINDOW_STATUS�v�Ŏn�܂���̂�ł��B

The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
Conditions are those beginning with "GLPA_WINDOW_STATUS" at line 28 of file "window.h". 
*/
#define ERROR_GLPA_UPDATE_WINDOW NULL

/*
Png�t�@�C����ǂݍ��߂܂���ł����B�t�@�C���p�X���Ⴄ�Ǝv���܂��B
load�֐��̈����֐������t�@�C���p�X��ݒ肵�Ă��������B

Png file could not be loaded. The file path seems to be different.
Set the correct file path to the Load function argument.
*/
#define ERROR_PNG_LOAD NULL

/*
�V�[���̍쐬���s���܂���ł����B�V�[���̕`����@�̑I�����Ⴄ�Ǝv���܂��B
�uscene.h�v�t�@�C���́uGLPA_SCENE�v�Ŏn�܂���̂�ł��B

Could not create the scene. It seems that you have chosen a different method of drawing the scene.
They are those starting with "GLPA_SCENE" in the "scene.h" file.
*/
#define ERROR_SCENE_CREATE NULL

/*
�V�[���t�H���_�[���ɂ���A�V�[�����Ƃ̃f�[�^���i�[����t�H���_�[�ւ̃p�X���������Ȃ��Ǝv���܂��B
�V�[���f�[�^���܂Ƃ߂�t�H���_�[���쐬���A���̒��ɃV�[�����Ƃ̃f�[�^���i�[����t�H���_�[���쐬���Ă��������B
�܂��A�V�[���f�[�^���܂Ƃ߂�t�H���_�[�̒��ɁA�摜�f�[�^�≹���f�[�^�Ȃǃt�H���_�[�ł͂Ȃ����͔z�u�����A�摜�f�[�^�Ȃǂ̓V�[�����Ƃ̃t�H���_�[�̒��ɔz�u���Ă��������B

The path to the folder in the scene folder that contains the data for each scene is probably incorrect.
Create a folder to organize scene data, and within that folder, create a folder to store data for each scene.
In addition, do not place non-folder items such as image data or sound data in folders that organize scene data, 
but place image data and other data in folders for each scene.
*/
#define ERROR_GLPA_LOAD_SCENE NULL

/*
2D�V�[���f�[�^�̉摜�̖��O�ɂ��̉摜��z�u����2�������W���܂܂�Ă��Ȃ��Ǝv���܂��B
�摜�t�@�C���̖��O�́u"�摜��"_@x"X���W"_@y"Y���W"�v�̂悤�ɂ��Ă��������B
PhotoShop�̏ꍇ�AGLPA�������ɂ���悤�ɃX�N���v�g���g�p���ăV�[���f�[�^���o�͂��邱�Ƃ��ł��܂��B

It is likely that the name of the image in the 2d scene data does not include the 2d coordinates 
in which the image is placed.
The name of the image file should be something like ""image name" @x "x-coordinates" @y "y-coordinates"".
In Photo shop, you can also use scripts to output scene data as described in the glpa manual.
*/
#define ERROR_SCENE2D_LOADPNG NULL

#endif ERROR_H_