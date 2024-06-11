#ifndef GLPA_SCENE_2D_H_
#define GLPA_SCENE_2D_H_

#include "Scene.h"
#include <algorithm>

namespace Glpa 
{

class Scene2d : public Scene
{
private :
    bool edited = true;

    Glpa::Render2d rend;

protected :
    /// @brief The name of the object is placed in the key value of the drawing order.
    std::unordered_map<int, std::vector<std::string>> drawOrder;

public :
    ~Scene2d() override;

    void setDrawOrder();
    void addDrawOrder(Glpa::SceneObject* obj);
    void deleteDrawOrder(Glpa::SceneObject* obj);

    void load() override;
    void release() override;

    void rendering(HDC dc,LPDWORD buf) override;

    bool getEdited() const {return edited;}
    void setEdited(bool symbol) {edited = symbol;}
    
};

}

#endif GLPA_SCENE_2D_H_
