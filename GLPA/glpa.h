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

class Glpa
{
public :
    void initialize(
        _In_ HINSTANCE arghInstance, _In_opt_ HINSTANCE arghPrevInstance, 
        _In_ LPSTR arglpCmdLine, _In_ int argnCmdShow
    );

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
        bool single_existence
    );

    void updateWindow(LPCWSTR window_name, int param);
    void setSingleWindow(bool single);
    bool dataSingleWindow();
    void deleteWindow();

    void runGraphicLoop();

    void createScene(std::string scene_name, int select_type);
    void loadScene(std::string scene_name, LPCWSTR scene_folder_path);
    void releaseScene(std::string scene_name);

    Scene2d* getPtScene2d(std::string scene_name);
    Scene3d* getPtScene3d(std::string scene_name);

    void setUserInputFunc(
        std::string scene_name, 
        std::wstring func_name, 
        GLPA_USER_FUNC_FUNCTIONAL pt_add_func,
        int message_type
    );
    void editUserInputFunc(std::wstring func_name, GLPA_USER_FUNC_FUNCTIONAL edited_func);
    void releaseUserInputFunc(std::wstring func_name);

    void setSceneActionFunc();

    void setSceneFrameFunc(std::string scene_name, std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL add_func);
    void editSceneFrameFunc(std::string scene_name, std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL edited_fuc);
    void releaseSceneFrameFunc(std::string scene_name, std::wstring func_name);

    void selectUseScene(LPCWSTR target_window_name, std::string scene_name);

    void selectUseCamera();

    void editImageInfo();

    void editCameraInfo();

    void editObjectInfo();

    void editCharacterInfo();

    MSG msg;

    std::unordered_map<LPCWSTR, Window> window;
    std::unordered_map<HWND, LPCWSTR> wndNames;

    std::unordered_map<std::string, HWND> scSetWnd;

    UserInput userInput;

private :
    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;
    GLPA_WINDOW_PROC_TYPE* ptWindowProc;

    bool singleWindow = false;

    Scene scene;

};

extern Glpa glpa;

LRESULT CALLBACK WindowProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

#endif  GLPA_H_
