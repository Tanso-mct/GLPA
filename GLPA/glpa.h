/**
 * @file glpa.h
 * @brief
 * 日本語 : ２次元及び３次元を連続で描画するライブラリの最上位ヘッダーファイル。
 * このファイルをインクルードし同じディレクトリにライブラリで使用するすべてのファイルを配置することでGLPAライブラリを使用できる。
 * 
 * English : Top-level header file for libraries that draw 2D and 3D in sequence.
 * You can use the glpa library by including this file and placing all the files used in the library 
 * in the same directory.
 * @author Tanso
 * @date 2023-7
*/


#ifndef GLPA_H_
#define GLPA_H_

#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "window.h"
#include "scene.h"
#include "error.h"
#include "user_input.h"

/**
 * 日本語 : GLPAライブラリの機能の格納する。
 * English : Store Glpa library functions.
*/
class Glpa{
public :
    /// @brief Initialize the glpa class, such as storing the arguments of the win main function when using glpa.
    /// @param arghInstance Handle to the current instance of the application.
    /// @param arghPrevInstance Handle to a previous instance of the application.
    /// @param arglpCmdLine The command line of the application, excluding the program name.
    /// @param argnCmdShow Controls how the window is displayed.
    void initialize(
        _In_ HINSTANCE arghInstance, _In_opt_ HINSTANCE arghPrevInstance, 
        _In_ LPSTR arglpCmdLine, _In_ int argnCmdShow
    );


    /// @brief Creates a new window from the argument and adds it to the variable that stores the window data.
    /// @param window_name Name of the window to be created.
    /// @param window_api_class_name Name of the window class in window api.
    /// @param window_width Window width.
    /// @param window_height Window height.
    /// @param window_dpi Window resolution.
    /// @param window_max_fps Maximum fps value for drawing in window.
    /// @param window_style Display style for window class registration.
    /// @param load_icon Icon for window class registration.
    /// @param load_cursor Cursor for window class registration.
    /// @param background_color The color of the window before the pixel is drawn, not the pixel color.
    /// @param small_icon Small icon for window class registration.
    /// @param minimize_auto Whether to automatically minimize this window when another window you have created is selected.
    /// @param single_existence Whether to minimize all other windows when this window is selected.
    /// @param wndViewStyle Window display style.
    void createWindow(
        LPCWSTR window_name,
        LPCWSTR window_api_class_name,
        int window_width,
        int window_height,
        int window_dpi,
        double window_max_fps,
        UINT window_style,
        LPWSTR load_icon, 
        LPWSTR load_cursor,
        int background_color,
        LPWSTR small_icon,
        bool minimize_auto,
        bool single_existence,
        DWORD wndViewStyle
    );


    /// @brief Change and update window information.
    /// @param window_name Name of symmetrical window.
    /// @param param Specify how the information is to be changed.

    /// @brief ウィンドウ情報を変更し、更新する。
    /// @param window_name 対称のウィンドウの名前。
    /// @param param 情報の変更の仕方を指定。
    void updateWindow(LPCWSTR window_name, int param);


    /// @brief Determines whether the mode should display only one window.
    /// @param single Whether or not to go into single-window mode.

    /// @brief 一つのウィンドウのみを表示するモードにするかどうかを決定する。
    /// @param single シングルウィンドウモードにするかどうか。
    void setSingleWindow(bool single);

    void releaseWindow();


    /// @brief Start the drawing loop.

    /// @brief 描画ループを開始する。
    void runGraphicLoop();


    /// @brief Execute the create function of the scene class.
    /// @param scene_name Name of the class to be created.
    /// @param select_type Specify whether it is 2-dimensional or 3-dimensional.

    /// @brief シーンクラスのcreate関数を実行する。
    /// @param scene_name 作成するクラスの名前。
    /// @param select_type ２次元または３次元かを指定する。
    void createScene(std::string scene_name, int select_type);


    /// @brief 
    /// @param scene_name 
    /// @param scene_folder_path 
    void loadScene(std::string scene_name, LPCWSTR scene_folder_path);
    void releaseScene(std::string scene_name);

    void selectUseScene(LPCWSTR target_window_name, std::string scene_name);

    Scene2d* getPtScene2d(std::string scene_name);
    Scene3d* getPtScene3d(std::string scene_name);

    void setUserInputFunc(
        std::string scene_name, std::wstring func_name, 
        GLPA_USER_FUNC_FUNCTIONAL pt_add_func,int message_type
    );
    void editUserInputFunc(std::wstring func_name, GLPA_USER_FUNC_FUNCTIONAL edited_func);
    void releaseUserInputFunc(std::wstring func_name);

    void setSceneFrameFunc(std::string scene_name, std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL add_func);
    void editSceneFrameFunc(std::string scene_name, std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL edited_fuc);
    void releaseSceneFrameFunc(std::string scene_name, std::wstring func_name);

    void selectUseCamera();

    void editImageInfo();

    void editCameraInfo();

    void editObjectInfo();

    void editCharacterInfo();

    MSG msg;

    std::unordered_map<LPCWSTR, Window> window;
    std::unordered_map<HWND, LPCWSTR> wndNames;

    bool singleWindow = false;
    int existWndAmount = 0;

    std::unordered_map<std::string, HWND> scSetWnd;

    UserInput userInput;

private :
    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;
    GLPA_WINDOW_PROC_TYPE* ptWindowProc;

    Scene scene;

};

extern Glpa glpa;

LRESULT CALLBACK WindowProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

#endif  GLPA_H_
