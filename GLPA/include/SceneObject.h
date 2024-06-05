#ifndef GLPA_SCENE_OBJECT_H_
#define GLPA_SCENE_OBJECT_H_

#include <string>

#include "Vector.h"

namespace Glpa
{

class SceneObject
{
protected :
    std::string name;

public :
    virtual ~SceneObject(){};

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}
    
    virtual void load() = 0;
    virtual void release() = 0;
};

}

#endif GLPA_SCENE_OBJECT_H_
