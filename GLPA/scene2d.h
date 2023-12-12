#ifndef SCENE2D_H_
#define SCENE2D_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "user_input.h"
#include "image.h"
#include "error.h"

#define GLPA_SCENE2D_FILENAME_X "_@x"
#define GLPA_SCENE2D_FILENAME_X_SIZE 3
#define GLPA_SCENE2D_FILENAME_Y "_@y"
#define GLPA_SCENE2D_FILENAME_Y_SIZE 3

class Scene2d
{
public :
    void loadPng(std::string folder_path, std::string group_name, std::string file_name);

private :
    UserInput userInput;
    std::unordered_map<std::string, std::vector<std::string>> group;
    std::unordered_map<std::string, Image> pngAttribute;

};

#endif  SCENE2D_H_