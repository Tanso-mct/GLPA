#ifndef SCENE2D_H_
#define SCENE2D_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "user_input.h"
#include "image.h"

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