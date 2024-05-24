#ifndef GLPA_SCENE_H_
#define GLPA_SCENE_H_

#include <string>

#include "WindowsInput.h"

namespace Glpa{

class Scene
{
private :
    std::string name;
    Glpa::WindowsInput* input;

public :
    virtual void start() = 0;
    virtual void update() = 0;

};

}

#endif GLPA_SCENE_H_
