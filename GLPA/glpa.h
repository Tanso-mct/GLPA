#ifndef GLPA_H_
#define GLPA_H_

#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "window.h"

#define ERROR_ARUGUMENT_INCOLLECT "GLPA ERROR : Argument is incorrect."

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

    void createScene();

    void loadScene();

    void setSceneUserInputFunc();

    void setSceneActionFunc();

    void setSceneFrameFunc();

    void selectUseScene();

    void selectUseCamera();

    void inputCameraInfo();

    void inputObjectInfo();

    void inputCharacterInfo();

    MSG msg;

    std::unordered_map<LPCWSTR, Window> window;

private :
    _In_ HINSTANCE hInstance;
    _In_opt_ HINSTANCE hPrevInstance;
    _In_ LPSTR lpCmdLine;
    _In_ int nCmdShow;
    WINDOW_PROC_TYPE* ptWindowProc;

    bool singleWindow = false;

};

extern Glpa glpa;

LRESULT CALLBACK windowProc(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

#endif  GLPA_H_
