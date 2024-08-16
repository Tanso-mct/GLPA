#ifndef GLPA_SCENE_3D_H_
#define GLPA_SCENE_3D_H_

#include "Scene.h"
#include "Material.h"

namespace Glpa 
{

class Scene3d : public Scene
{
private :
    Glpa::Render3d rend;
    std::unordered_map<std::string, Glpa::Material*> mts;

protected :
    
public :
    Scene3d();
    ~Scene3d() override;

    void load() override;
    void release() override;

    void AddMaterial(Glpa::Material* ptMt);
    void DeleteMaterial(Glpa::Material* ptMt);

    void rendering
    (
        ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, HWND hWnd, PAINTSTRUCT ps,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
    ) override;

};

}

#endif GLPA_SCENE_3D_H_
