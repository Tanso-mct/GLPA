#ifndef SCENE2D_H_
#define SCENE2D_H_

#include <unordered_map>
#include <string>

#include "user_input.h"

class Scene2d
{
public :
    bool loadPng();

private :
    UserInput userInput;
    std::unordered_map<std::string, std::string> imageAttribute;
};

#endif  SCENE2D_H_