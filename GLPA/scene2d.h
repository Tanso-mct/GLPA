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

#define GLPA_SCENE_FUNC_FUNCTIONAL std::function<void(HDC, LPDWORD, int, int, int)>

#define GLPA_SCENE_FUNC_PT(instance, method_name) \
    [&instance](HDC hBufDC, LPDWORD lpPixel, int width, int height, int dpi){ \
        instance.method_name(hBufDC, lpPixel, width, height, dpi); \
    }

#define GLPA_SCENE_FUNC(method_name) \
    void method_name(HDC hBufDC, LPDWORD lpPixel, int width, int height, int dpi)

#define GLPA_USER_FUNC_DEFINE(class_name, method_name, h_buffer_dc_arg_name, lp_pixel_arg_name, width_arg_name, height_arg_name, dpi_arg_name) \
    void class_name::method_name(HDC h_buffer_dc_arg_name, LPDWORD lp_pixel_arg_name, int width_arg_name, int height_arg_name, int dpi_arg_name)


class Scene2d
{
public :
    void loadPng(std::string folder_path, std::string group_name, std::string file_name);
    void loadText();
    void reload();
    void release();

    void showImage();
    void showGroup();

    void edit(HDC h_buffer_dc, LPDWORD lp_pixel, int width, int height, int dpi);

    void update(
        HDC h_buffer_dc,
        LPDWORD window_buffer,
        int window_width,
        int window_height,
        int window_dpi
    );

    std::unordered_map<std::string, int> groupOrder;
    std::unordered_map<int, std::unordered_map<int, std::string>> layerOrder;

    Text text;

    bool edited = true;

    void addSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL add_func);
    void editSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL edited_func);
    void releaseSceneFrameFunc(std::wstring func_name);

private :
    std::unordered_map<std::string, std::vector<std::string>> group;
    std::unordered_map<std::string, Image> pngAttribute;

    std::unordered_map<std::wstring, GLPA_SCENE_FUNC_FUNCTIONAL> sceneFrameFunc;

    Color color;

};

#endif  SCENE2D_H_