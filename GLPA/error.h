/*
GLPAエラー説明書

Visual studioでの開発の場合、エラー箇所のマクロをCtrl + クリックでこのファイルへ移動します。
各エラーの原因を記述しています。指示に従い修正を行ってください。

また、開発者は日本人なためまず日本語で記述しそれをDeepLで翻訳しています。
翻訳内容の確認は開発者本人が行いましたが、英語が完全ではないため誤訳が存在する可能性があります。注意してください。

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
引数の「LPCWSTR wndName」の値がスイッチ文の条件と一致しませんでした。
何の処理も行いませんでした。引数の値をスイッチ文の条件のいずれかにしてください。
条件は「window.h」ファイルの「GLPA_WINDOW_STATUS」で始まるものらです。

The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
Conditions are those beginning with "GLPA_WINDOW_STATUS" at line 28 of file "window.h". 
*/
#define ERROR_GLPA_UPDATE_WINDOW NULL

/*
Pngファイルを読み込めませんでした。ファイルパスが違うと思われます。
load関数の引数へ正しいファイルパスを設定してください。

Png file could not be loaded. The file path seems to be different.
Set the correct file path to the Load function argument.
*/
#define ERROR_PNG_LOAD NULL

/*
シーンの作成を行えませんでした。シーンの描画方法の選択が違うと思われます。
「scene.h」ファイルの「GLPA_SCENE」で始まるものらです。

Could not create the scene. It seems that you have chosen a different method of drawing the scene.
They are those starting with "GLPA_SCENE" in the "scene.h" file.
*/
#define ERROR_SCENE_CREATE NULL

/*
シーンフォルダー内にある、シーンごとのデータを格納するフォルダーへのパスが正しくないと思われます。
シーンデータをまとめるフォルダーを作成し、その中にシーンごとのデータを格納するフォルダーを作成してください。
また、シーンデータをまとめるフォルダーの中に、画像データや音声データなどフォルダーではない物は配置せず、画像データなどはシーンごとのフォルダーの中に配置してください。

The path to the folder in the scene folder that contains the data for each scene is probably incorrect.
Create a folder to organize scene data, and within that folder, create a folder to store data for each scene.
In addition, do not place non-folder items such as image data or sound data in folders that organize scene data, 
but place image data and other data in folders for each scene.
*/
#define ERROR_GLPA_LOAD_SCENE NULL

/*
2Dシーンデータの画像の名前にその画像を配置する2次元座標が含まれていないと思われます。
画像ファイルの名前は「"画像名"_@x"X座標"_@y"Y座標"」のようにしてください。
PhotoShopの場合、GLPA説明書にあるようにスクリプトを使用してシーンデータを出力することもできます。

It is likely that the name of the image in the 2d scene data does not include the 2d coordinates 
in which the image is placed.
The name of the image file should be something like ""image name" @x "x-coordinates" @y "y-coordinates"".
In Photo shop, you can also use scripts to output scene data as described in the glpa manual.
*/
#define ERROR_SCENE2D_LOADPNG NULL

#endif ERROR_H_