#ifndef SCENE2D_H_
#define SCENE2D_H_

#include <unordered_map>
#include <string>
#include <vector>

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



class Scene2d
{
public :
    void loadPng(std::string folder_path, std::string group_name, std::string file_name);
    void reload();
    void release();

    void showImage();
    void showGroup();

    void edit();

    void update(
        HDC h_buffer_dc,
        LPDWORD window_buffer,
        int window_width,
        int window_height,
        int window_dpi
    );

    std::unordered_map<std::string, int> groupOrder;
    std::unordered_map<int, std::unordered_map<int, std::string>> layerOrder;

private :
    bool edited = true;
    std::unordered_map<std::string, std::vector<std::string>> group;
    std::unordered_map<std::string, Image> pngAttribute;

    Color color;
    Text text;

};

#endif  SCENE2D_H_