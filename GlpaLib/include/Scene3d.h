#ifndef GLPA_SCENE_3D_H_
#define GLPA_SCENE_3D_H_

#include "Scene.h"

namespace Glpa 
{

class Scene3d : public Scene
{
private :
    Glpa::Render3d rend;

protected :
    
public :
    ~Scene3d() override;

    void load() override;
    void release() override;

    void rendering(LPDWORD& buf, int& bufWidth, int& bufHeight, int& bufDpi) override;

};

}

#endif GLPA_SCENE_3D_H_
