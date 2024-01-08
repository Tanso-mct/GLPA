/**
 * @file error.h
 * @brief
 * ���{�� : GLPA�ŃG���[���N�������ۂɗ��R���m�F���邽�߂Ɏg�p����B
 * English : Used to check the reason when an error occurs in Glpa.
 * @author Tanso
 * @date 2023-10
*/

/**********************************************************************************************************************
 * ���{�� : GLPA�G���[������
 * Visual studio�ł̊J���̏ꍇ�A�G���[�ӏ��̃}�N����Ctrl + �N���b�N�ł��̃t�@�C���ֈړ����܂��B
 * �e�G���[�̌������L�q���Ă��܂��B�w���ɏ]���C�����s���Ă��������B
 * 
 * �܂��A�J���҂͓��{�l�Ȃ��߂܂����{��ŋL�q�������DeepL�Ŗ|�󂵂Ă��܂��B
 * �|����e�̊m�F�͊J���Җ{�l���s���܂������A�p�ꂪ���S�ł͂Ȃ����ߌ�󂪑��݂���\��������܂��B���ӂ��Ă��������B
 * 
 * 
 * English : GLPA Error Instructions
 * For development in Visual studio, Ctrl + click on the macro in the error location to move to this file. 
 * The causes of each error are described. Please follow the instructions to correct the errors.
 * 
 * The developer is Japanese, so the text is first written in Japanese, and then translated by DeepL.
 * The translations have been checked by the developer himself, however, 
 * the English is not perfect and mistranslations may exist. Please be careful.
**********************************************************************************************************************/


#ifndef ERROR_H_
#define ERROR_H_

#include <stdexcept>


/**********************************************************************************************************************
 * ���{�� : �����́uLPCWSTR wndName�v�̒l���X�C�b�`���̏����ƈ�v���܂���ł����B
 * ���̏������s���܂���ł����B�����̒l���X�C�b�`���̏����̂����ꂩ�ɂ��Ă��������B
 * �����́uwindow.h�v�t�@�C���́uGLPA_WINDOW_STATUS�v�Ŏn�܂���̂�ł��B
 * 
 * English : The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
 * No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
 * Conditions are those beginning with "GLPA_WINDOW_STATUS" at line 28 of file "window.h".
**********************************************************************************************************************/
#define ERROR_GLPA_UPDATE_WINDOW NULL



/**********************************************************************************************************************
 * ���{�� : Png�t�@�C����ǂݍ��߂܂���ł����B�t�@�C���p�X���Ⴄ�Ǝv���܂��B
 * load�֐��̈����֐������t�@�C���p�X��ݒ肵�Ă��������B
 * 
 * English : Png file could not be loaded. The file path seems to be different.
 * Set the correct file path to the Load function argument.
**********************************************************************************************************************/
#define ERROR_PNG_LOAD NULL



/**********************************************************************************************************************
 * ���{�� : �V�[���̍쐬���s���܂���ł����B�V�[���̕`����@�̑I�����Ⴄ�Ǝv���܂��B
 * �uscene.h�v�t�@�C���́uGLPA_SCENE�v�Ŏn�܂���̂�ł��B
 * 
 * English : Could not create the scene. It seems that you have chosen a different method of drawing the scene.
 * They are those starting with "GLPA_SCENE" in the "scene.h" file.
**********************************************************************************************************************/
#define ERROR_SCENE_CREATE NULL

/*
�V�[���t�H���_�[���ɂ���A�V�[�����Ƃ̃f�[�^���i�[����t�H���_�[�ւ̃p�X���������Ȃ��Ǝv���܂��B
�V�[���f�[�^���܂Ƃ߂�t�H���_�[���쐬���A���̒��ɃV�[�����Ƃ̃f�[�^���i�[����t�H���_�[���쐬���Ă��������B
�܂��A�V�[���f�[�^���܂Ƃ߂�t�H���_�[�̒��ɁA�摜�f�[�^�≹���f�[�^�Ȃǃt�H���_�[�ł͂Ȃ����͔z�u�����A
�摜�f�[�^�Ȃǂ̓V�[�����Ƃ̃t�H���_�[�̒��ɔz�u���Ă��������B

The path to the folder in the scene folder that contains the data for each scene is probably incorrect.
Create a folder to organize scene data, and within that folder, create a folder to store data for each scene.
In addition, do not place non-folder items such as image data or sound data in folders that organize scene data, 
but place image data and other data in folders for each scene.
*/
#define ERROR_GLPA_LOAD_SCENE NULL

/*
2D�V�[���f�[�^�̉摜�̖��O�ɂ��̉摜��z�u����2�������W���܂܂�Ă��Ȃ��Ǝv���܂��B
�摜�t�@�C���̖��O�́u"�摜��"_@x"X���W"_@y"Y���W_@l"���C���[�ԍ�(��ԏ�̃��C���[���P�Ƃ��ĉ��֏���)"�v�̂悤�ɂ��Ă��������B
PhotoShop�̏ꍇ�AGLPA�������ɂ���悤�ɃX�N���v�g���g�p���ăV�[���f�[�^���o�͂��邱�Ƃ��ł��܂��B

It is likely that the name of the image in the 2d scene data does not include the 2d coordinates 
in which the image is placed.
The name of the image file should be ""Image Name" @x "x-coordinates" @y "y-coordinates @l "layer number 
(ascending order with the top layer as 1)".
In Photo shop, you can also use scripts to output scene data as described in the glpa manual.
*/
#define ERROR_SCENE2D_LOADPNG NULL

/*
Window�N���X�̍쐬�Ɏ��s���܂����B
GLPA::createWindow�֐��̈������Ԉ���Ă���\��������܂��B�������͎��s���ɂ���肪����\��������܂��B

Failed to create Window class.
The argument of the Glpa::createWindow function may be incorrect. 
Or there may be a problem with the execution environment as well.
*/
#define ERROR_WINDOW_REGISTER_CLASS NULL

/*
�E�B���h�E�̍쐬�Ɏ��s���܂����B
GLPA::createWindow�֐��̈������Ԉ���Ă���\��������܂��B�������͎��s���ɂ���肪����\��������܂��B

Failed to create Window.
The argument of the Glpa::createWindow function may be incorrect. 
Or there may be a problem with the execution environment as well.
*/
#define ERROR_WINDOW_CREATE NULL

/*
�ҏW�^�C�v�����݂��܂���B
�����ł̎w��Ɍ�肪����\��������܂���B�uGLPA_TEXT_EDIT�v�Ŏn�܂�}�N���̂����ꂩ��I�����Ă��������B

Edit type does not exist.
There is no possibility of an incorrect specification in the argument. 
Choose from any of the macros beginning with "glpa text edit".
*/
#define ERROR_TEXT_EDIT NULL

/*
���b�Z�[�W����������֐���ǉ����邽�߂ɕK�v�ȃ��b�Z�[�W�^�C�v�̎w�肪����Ă���\��������܂��B
�uGLPA_USERINPUT_MESSAGE_�v�Ŏn�܂邢���ꂩ�̃}�N����I�����A�����Ɏw�肵�Ă��������B

You may have incorrectly specified the message type needed to add a function to process the message.
Select one of the macros beginning with "GLPA_USERINPUT_MESSAGE_" and specify it as an argument.
*/
#define ERROR_USER_INPUT_ADD NULL

/*
����܂ō쐬���ꂽ�V�[��2D�f�[�^�̒��Ɉ����Ŏw�肵���V�[���̖��O�����V�[�������݂��܂���B
�K�؂Ɉ����ɃV�[���̖��O����͂��Ă��������B

There is no scene with the name of the scene specified in the argument in the scene2d data created so far.
Enter the name of the scene in the appropriate argument.
*/
#define ERROR_GLPA_GET_PT_SCENE2D NULL

/*
����܂ō쐬���ꂽ�V�[��3D�f�[�^�̒��Ɉ����Ŏw�肵���V�[���̖��O�����V�[�������݂��܂���B
�K�؂Ɉ����ɃV�[���̖��O����͂��Ă��������B

There is no scene with the name of the scene specified in the argument in the scene3d data created so far.
Enter the name of the scene in the appropriate argument.
*/
#define ERROR_GLPA_GET_PT_SCENE3D NULL

/*
�֐��̈����Ŏw�肵�Ă��閼�O�̊֐������݂��܂���B�K�؂Ɋ֐��̈����ɒl����͂��Ă��������B

The function with the name specified in the function argument does not exist. 
Please enter a value for the function argument appropriately.
*/
#define ERROR_GLPA_SCENE_2D_FRAME_FUNC_NAME NULL


/*
�֐��̈����Ŏw�肵�Ă��閼�O�̊֐������݂��܂���B�K�؂Ɋ֐��̈����ɒl����͂��Ă��������B

The function with the name specified in the function argument does not exist. 
Please enter a value for the function argument appropriately.
*/
#define ERROR_GLPA_SCENE_3D_FRAME_FUNC_NAME NULL


/**********************************************************************************************************************
 * ���{�� : mesh.h�y��mesh.cpp�t�@�C���Ɋւ���G���[���b�Z�[�W�ꗗ�B
 * English : List of error messages related to mesh.h and mesh.cpp files.
**********************************************************************************************************************/


/********************************************************************************
 * ���{�� : �t�@�C���̓ǂݍ��݂Ɏ��s���܂����B
 * �����ɓK�؂Ƀt�@�C���̖��O�ƁA�t�@�C�������݂���t�H���_�[�̃p�X���w�肵�Ă��������B
 * 
 * English : Failed to load file. Please specify the name of the file and 
 * the path to the folder where the file resides in the argument appropriately.
********************************************************************************/
#define ERROR_MESH_LOAD_FILE NULL


/********************************************************************************
 * ���{�� : �����Ŏw�肵�A������悤�Ƃ������̂����݂��܂���B
 * �K�؂ɉ������Ώۂ̃t�@�C�����������Ɏw�肵�Ă��������B
 * 
 * English : Specified by argument, the file you attempted to free does not exist.
 * Specify the name of the file to be properly released in the argument.
********************************************************************************/
#define ERROR_MESH_LOAD_RELEASE NULL


/**********************************************************************************************************************
 * ���{�� : command.h�y��command.cpp�t�@�C���Ɋւ���G���[���b�Z�[�W�ꗗ
 * English : List of error messages related to the command.h and command.cpp files
**********************************************************************************************************************/


/********************************************************************************
 * ���{�� : �����Ŏw�肵�����O�̊֐������݂��܂���B
 * �K�؂Ɋ֐��̖��O�������֎w�肵�Ă��������B
 * 
 * English : The function with the name specified in the argument does not exist.
 * Specify an appropriate function name for the argument.
********************************************************************************/
#define ERROR_COMMAND_LIST NULL


#endif ERROR_H_