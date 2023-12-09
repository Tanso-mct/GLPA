#ifndef SCENE_H_
#define SCENE_H_

#include <string>
#include <unordered_map>

#include "scene2d.h"
#include "scene3d.h"

#define GLPA_SCENE_2D 0
#define GLPA_SCENE_3D 1

class Scene
{
public :
    void create();
    void load();
    void release();
    void reload();
    void remove();
    void update();

private :
    std::unordered_map<std::string, int> names;
    std::unordered_map<std::string, Scene2d> data2d;
    std::unordered_map<std::string, Scene3d> data3d;

};

#endif  SCENE_H_

