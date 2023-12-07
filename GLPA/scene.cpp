#include "scene.h"

void Scene::create(std::string scrName, int graphicType){
    if (graphicType == GRAPHIC_2D){
        Scene2d newScene2d;
        data2d.emplace(scrName, newScene2d);
    }
    else if(graphicType == GRAPHIC_3D){
        Scene3d newScene3d;
        data3d.emplace(scrName, newScene3d);
    }
}
