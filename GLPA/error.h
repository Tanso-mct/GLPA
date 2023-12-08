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
条件は「window.h」ファイルの28行目の「WINDOW_STATUS」で始まるものらです。

The value of the argument "LPCWSTR wndName" did not match the condition of the switch statement.
No processing was performed. The value of the argument should be one of the conditions of the switch statement. 
Conditions are those beginning with "WINDOW_STATUS" at line 28 of file "window.h". 
*/
#define ERROR_GLPA_UPDATE_WINDOW NULL

/*
Pngファイルを読み込めませんでした。ファイルパスが違うと思われます。
load関数の引数へ正しいファイルパスを設定してください。

Png file could not be loaded. The file path seems to be different.
Set the correct file path to the Load function argument.
*/
#define ERROR_PNG_LOAD NULL


#endif ERROR_Hでで