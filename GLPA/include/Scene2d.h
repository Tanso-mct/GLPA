#ifndef GLPA_SCENE_2D_H_
#define GLPA_SCENE_2D_H_

#include "Scene.h"

namespace Glpa 
{

class Scene2d : public Scene
{
protected :
    /// @brief The name of the object is placed in the key value of the drawing order.
    std::unordered_map<int, std::vector<std::string>> drawOrder;

public :
    ~Scene2d() override;

    void setDrawOrder();
    
};

}

#endif GLPA_SCENE_2D_H_
