#ifndef SCENE_H_
#define SCENE_H_

#include <string>
#include <unordered_map>

#include "scene2d.h"
#include "scene3d.h"

class Scene
{
public :
    void create();
    void load();
    void 
    void remove();

private :
    std::unordered_map<std::string, int> names;
    std::unordered_map<std::string, Scene2d> data2d;
    std::unordered_map<std::string, Scene3d> data3d;

};

#endif  SCENE_H_

