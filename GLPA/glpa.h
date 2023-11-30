#ifndef GLPA_H_
#define GLPA_H_

#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>

#include "window_api.h"

class Glpa
{
public :
    Glpa
    (
        _In_ HINSTANCE arghInstance, _In_opt_ HINSTANCE arghPrevInstance, 
        _In_ LPSTR arglpCmdLine, _In_ int argnCmdShow
    )
    {
        windowApi.hInstance = arghInstance;
        windowApi.hPrevInstance = arghPrevInstance;
        windowApi.lpCmdLine = arglpCmdLine;
        windowApi.nCmdShow = argnCmdShow;
    }

    void createWindow
    (
        std::string window_name,
        std::string window_api_class_name,
        double window_width,
        double window_height,
        double window_dpi,
        double window_max_fps,
        bool window_full_screen,
        UINT window_style,
        LPWSTR load_icon, 
        LPWSTR load_cursor,
        int background_color,
        LPWSTR small_icon
    );

    void showWindow(std::string window_name);
    
    void updateWindowInfo();

    void deleteWindow();

    void graphicLoop();

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

private :
    WindowApi windowApi;
    std::unordered_map<std::string, Window> window;

};

#endif  GLPA_H_
