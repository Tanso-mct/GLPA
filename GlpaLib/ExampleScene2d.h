#ifndef EXAMPLE_SCENE_2D_H_
#define EXAMPLE_SCENE_2D_H_

#pragma warning(disable : 4996)

#include "Scene2d.h"

#include <string>
#include <Windows.h>

class ExampleScene2d : public Glpa::Scene2d
{
private :
    bool opened = false;
    Glpa::Image* ptBackGround;

    Glpa::Vec2d beforeImgPos;
    Glpa::Vec2d mouseLDownPos;
    bool isImgMoving = false;

public :
    ~ExampleScene2d() override;

    void openExample3d();
    
    void setup() override;

    void start() override;

    void update() override;
};

#endif EXAMPLE_SCENE_2D_H_
