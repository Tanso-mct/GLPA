#ifndef GLPA_SCENE_2D_H_
#define GLPA_SCENE_2D_H_

#include "Scene.h"

#include <algorithm>
#include <map>

#pragma comment(lib, "d2d1")

namespace Glpa 
{

class Scene2d : public Scene
{
private :
    int imgAmount = 0;
    int textAmount = 0;

    Glpa::Render2d rend;

protected :
    /// @brief The name of the object is placed in the key value of the drawing order.
    std::map<int, std::vector<std::string>> drawOrder;

public :
    Scene2d();
    ~Scene2d() override;

    void editPos(Glpa::Image* img, Glpa::Vec2d newPos);
    void EditDrawOrder(Glpa::SceneObject *obj, int newDrawOrder);

    void addDrawOrder(Glpa::SceneObject* obj);
    void deleteDrawOrder(Glpa::SceneObject* obj);

    void load() override;
    void release() override;

    void rendering(ID2D1HwndRenderTarget* pRenderTarget, ID2D1Bitmap** pBitMap, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi) override;

};

}

#endif GLPA_SCENE_2D_H_
