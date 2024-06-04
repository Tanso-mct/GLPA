#include "ExampleScene2d.h"

void ExampleScene2d::setup()
{
    ptBackGround = new Glpa::Image("back_ground","resource/Assets/Images/console_back_ground.png", Glpa::Vec2d(0, 0));

    AddSceneObject(ptBackGround);
}