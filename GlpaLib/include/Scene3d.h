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
    Scene3d();
    ~Scene3d() override;

    void load() override;
    void release() override;

    void rendering(ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi) override;

};

}

#endif GLPA_SCENE_3D_H_
