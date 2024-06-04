#ifndef GLPA_SCENE_2D_H_
#define GLPA_SCENE_2D_H_

#include "Scene.h"

namespace Glpa 
{

class Scene2d : public Scene
{
private :

public :
    ~Scene2d() override;
    
    void load() override;
    void release() override;

};

}

#endif GLPA_SCENE_2D_H_
