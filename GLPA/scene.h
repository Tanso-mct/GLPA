#ifndef SCREEN_H_
#define SCREEN_H_

#include <string>
#include <unordered_map>

#include "scene2d.h"
#include "scene3d.h"

#define GRAPHIC_2D 0
#define GRAPHIC_3D 1

class Scene
{
public :
    void create(std::string scene_name, int graphic_type);
    void load();
    void free();
    void remove();
    void update();

private :
    std::unordered_map<std::string, int> names;
    std::unordered_map<std::string, Scene2d> data2d;
    std::unordered_map<std::string, Scene3d> data3d;
};

#endif SCREEN_H_


