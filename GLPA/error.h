/**
 * @file error.h
 * @brief
 * 日本語 : GLPAでエラーが起こった際に理由を確認するために使用する。
 * English : Used to check the reason when an error occurs in Glpa.
 * @author Tanso
 * @date 2023-10
*/

/**********************************************************************************************************************
 * 日本語 : GLPAエラー説明書
 * Visual studioでの開発の場合、エラー箇所のマクロをCtrl + クリックでこのファイルへ移動します。
 * 各エラーの原因を記述しています。指示に従い修正を行ってください。
 * 
 * また、開発者は日本人なためまず日本語で記述しそれをDeepLで翻訳しています。
 * 翻訳内容の確認は開発者本人が行いましたが、英語が完全ではないため誤訳が存在する可能性があります。注意してください。
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
 * 日本語 : 引数の「LPCWSTR wndName」の値がスイッチ文の条件と一致しませんでした。
 * 何の処理も行いませんでした。引数の値をスイッチ文の条件のいずれかにしてください。
 * 条件は「window.h」ファイルの「GLPA_WINDOW_STATUS」で始まるものらです。
 * 
 * English : The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
 * No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
 * Conditions are those beginning with "GLPA_WINDOW_STATUS" at line 28 of file "window.h".
**********************************************************************************************************************/
#define ERROR_GLPA_UPDATE_WINDOW NULL



/**********************************************************************************************************************
 * 日本語 : Pngファイルを読み込めませんでした。ファイルパスが違うと思われます。
 * load関数の引数へ正しいファイルパスを設定してください。
 * 
 * English : Png file could not be loaded. The file path seems to be different.
 * Set the correct file path to the Load function argument.
**********************************************************************************************************************/
#define ERROR_PNG_LOAD NULL



/**********************************************************************************************************************
 * 日本語 : シーンの作成を行えませんでした。シーンの描画方法の選択が違うと思われます。
 * 「scene.h」ファイルの「GLPA_SCENE」で始まるものらです。
 * 
 * English : Could not create the scene. It seems that you have chosen a different method of drawing the scene.
 * They are those starting with "GLPA_SCENE" in the "scene.h" file.
**********************************************************************************************************************/
#define ERROR_SCENE_CREATE NULL

/*
シーンフォルダー内にある、シーンごとのデータを格納するフォルダーへのパスが正しくないと思われます。
シーンデータをまとめるフォルダーを作成し、その中にシーンごとのデータを格納するフォルダーを作成してください。
また、シーンデータをまとめるフォルダーの中に、画像データや音声データなどフォルダーではない物は配置せず、
画像データなどはシーンごとのフォルダーの中に配置してください。

The path to the folder in the scene folder that contains the data for each scene is probably incorrect.
Create a folder to organize scene data, and within that folder, create a folder to store data for each scene.
In addition, do not place non-folder items such as image data or sound data in folders that organize scene data, 
but place image data and other data in folders for each scene.
*/
#define ERROR_GLPA_LOAD_SCENE NULL

/*
2Dシーンデータの画像の名前にその画像を配置する2次元座標が含まれていないと思われます。
画像ファイルの名前は「"画像名"_@x"X座標"_@y"Y座標_@l"レイヤー番号(一番上のレイヤーを１として下へ昇順)"」のようにしてください。
PhotoShopの場合、GLPA説明書にあるようにスクリプトを使用してシーンデータを出力することもできます。

It is likely that the name of the image in the 2d scene data does not include the 2d coordinates 
in which the image is placed.
The name of the image file should be ""Image Name" @x "x-coordinates" @y "y-coordinates @l "layer number 
(ascending order with the top layer as 1)".
In Photo shop, you can also use scripts to output scene data as described in the glpa manual.
*/
#define ERROR_SCENE2D_LOADPNG NULL

/*
Windowクラスの作成に失敗しました。
GLPA::createWindow関数の引数が間違っている可能性があります。もしくは実行環境にも問題がある可能性があります。

Failed to create Window class.
The argument of the Glpa::createWindow function may be incorrect. 
Or there may be a problem with the execution environment as well.
*/
#define ERROR_WINDOW_REGISTER_CLASS NULL

/*
ウィンドウの作成に失敗しました。
GLPA::createWindow関数の引数が間違っている可能性があります。もしくは実行環境にも問題がある可能性があります。

Failed to create Window.
The argument of the Glpa::createWindow function may be incorrect. 
Or there may be a problem with the execution environment as well.
*/
#define ERROR_WINDOW_CREATE NULL

/*
編集タイプが存在しません。
引数での指定に誤りがある可能性がありません。「GLPA_TEXT_EDIT」で始まるマクロのいずれから選択してください。

Edit type does not exist.
There is no possibility of an incorrect specification in the argument. 
Choose from any of the macros beginning with "glpa text edit".
*/
#define ERROR_TEXT_EDIT NULL

/*
メッセージを処理する関数を追加するために必要なメッセージタイプの指定が誤っている可能性があります。
「GLPA_USERINPUT_MESSAGE_」で始まるいずれかのマクロを選択し、引数に指定してください。

You may have incorrectly specified the message type needed to add a function to process the message.
Select one of the macros beginning with "GLPA_USERINPUT_MESSAGE_" and specify it as an argument.
*/
#define ERROR_USER_INPUT_ADD NULL

/*
これまで作成されたシーン2Dデータの中に引数で指定したシーンの名前を持つシーンが存在しません。
適切に引数にシーンの名前を入力してください。

There is no scene with the name of the scene specified in the argument in the scene2d data created so far.
Enter the name of the scene in the appropriate argument.
*/
#define ERROR_GLPA_GET_PT_SCENE2D NULL

/*
これまで作成されたシーン3Dデータの中に引数で指定したシーンの名前を持つシーンが存在しません。
適切に引数にシーンの名前を入力してください。

There is no scene with the name of the scene specified in the argument in the scene3d data created so far.
Enter the name of the scene in the appropriate argument.
*/
#define ERROR_GLPA_GET_PT_SCENE3D NULL

/*
関数の引数で指定している名前の関数が存在しません。適切に関数の引数に値を入力してください。

The function with the name specified in the function argument does not exist. 
Please enter a value for the function argument appropriately.
*/
#define ERROR_GLPA_SCENE_2D_FRAME_FUNC_NAME NULL


/*
関数の引数で指定している名前の関数が存在しません。適切に関数の引数に値を入力してください。

The function with the name specified in the function argument does not exist. 
Please enter a value for the function argument appropriately.
*/
#define ERROR_GLPA_SCENE_3D_FRAME_FUNC_NAME NULL


/**********************************************************************************************************************
 * 日本語 : mesh.h及びmesh.cppファイルに関するエラーメッセージ一覧。
 * English : List of error messages related to mesh.h and mesh.cpp files.
**********************************************************************************************************************/


/********************************************************************************
 * 日本語 : ファイルの読み込みに失敗しました。
 * 引数に適切にファイルの名前と、ファイルが存在するフォルダーのパスを指定してください。
 * 
 * English : Failed to load file. Please specify the name of the file and 
 * the path to the folder where the file resides in the argument appropriately.
********************************************************************************/
#define ERROR_MESH_LOAD_FILE NULL


/********************************************************************************
 * 日本語 : 引数で指定し、解放しようとしたものが存在しません。
 * 適切に解放する対象のファイル名を引数に指定してください。
 * 
 * English : Specified by argument, the file you attempted to free does not exist.
 * Specify the name of the file to be properly released in the argument.
********************************************************************************/
#define ERROR_MESH_LOAD_RELEASE NULL


/**********************************************************************************************************************
 * 日本語 : command.h及びcommand.cppファイルに関するエラーメッセージ一覧
 * English : List of error messages related to the command.h and command.cpp files
**********************************************************************************************************************/


/********************************************************************************
 * 日本語 : 引数で指定した名前の関数が存在しません。
 * 適切に関数の名前を引数へ指定してください。
 * 
 * English : The function with the name specified in the argument does not exist.
 * Specify an appropriate function name for the argument.
********************************************************************************/
#define ERROR_COMMAND_LIST NULL


#endif ERROR_H_