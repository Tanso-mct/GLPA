#ifndef EXAMPLE_SCENE_2D_H_
#define EXAMPLE_SCENE_2D_H_

#include "Scene2d.h"

class ExampleScene2d : public Glpa::Scene2d
{
private :
    bool opened = false;
    Glpa::Image* ptBackGround;

public :
    ~ExampleScene2d() override;

    void openExample3d();
    
    void setup() override;

    void update() override;
};

#endif EXAMPLE_SCENE_2D_H_
