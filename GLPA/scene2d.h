#ifndef SCENE2D_H_
#define SCENE2D_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>

#include "image.h"
#include "error.h"
#include "color.h"
#include "text.h"

#define GLPA_SCENE2D_FILENAME_X "_@x"
#define GLPA_SCENE2D_FILENAME_X_SIZE 3
#define GLPA_SCENE2D_FILENAME_Y "_@y"
#define GLPA_SCENE2D_FILENAME_Y_SIZE 3
#define GLPA_SCENE2D_FILENAME_L "_@l"
#define GLPA_SCENE2D_FILENAME_L_SIZE 3

#define GLPA_SCENE_FUNC_FUNCTIONAL std::function<void(HDC, LPDWORD)>

#define GLPA_SCENE_FUNC_PT(instance, method_name) \
    [&instance](HDC hBufDC, LPDWORD lpPixel){ \
        instance.method_name(hBufDC, lpPixel); \
    }

#define GLPA_SCENE_FUNC(method_name) \
    void method_name(HDC h_buffer_dc, LPDWORD lp_pixel)


class Scene2d
{
public :
    void storeUseWndParam(int width, int height, int dpi);

    void loadPng(std::string folder_path, std::string group_name, std::string file_name);
    void loadText();
    void reload();
    void release();

    void showImage();
    void showGroup();

    void edit(HDC h_buffer_dc, LPDWORD lp_pixel);

    void update(HDC h_buffer_dc, LPDWORD window_buffer);

    std::unordered_map<std::string, int> groupOrder;
    std::unordered_map<int, std::unordered_map<int, std::string>> layerOrder;

    Text text;

    bool edited = true;

    void addSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL add_func);
    void editSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL edited_func);
    void releaseSceneFrameFunc(std::wstring func_name);

    int useWndWidth = 0;
    int useWndHeight = 0;
    int useWndDpi = 0;

private :

    std::unordered_map<std::string, std::vector<std::string>> group;
    std::unordered_map<std::string, Image> pngAttribute;

    std::unordered_map<std::wstring, GLPA_SCENE_FUNC_FUNCTIONAL> sceneFrameFunc;

    Color color;

};

#endif  SCENE2D_H_