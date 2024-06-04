#ifndef GLPA_SCENE_3D_H_
#define GLPA_SCENE_3D_H_

#include "Scene.h"

namespace Glpa 
{

class Scene3d : public Scene
{
private :
    
public :
    ~Scene3d() override;

    void load() override;
    void release() override;
    
};

}

#endif GLPA_SCENE_3D_H_
