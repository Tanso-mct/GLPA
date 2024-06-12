#ifndef EXAMPLE_SCENE_2D_H_
#define EXAMPLE_SCENE_2D_H_

#pragma warning(disable : 4996)

#include "Scene2d.h"

#include <string>
#include <Windows.h>
#include <sapi.h>
#include <atlbase.h>
#include <sphelper.h> 

class ExampleScene2d : public Glpa::Scene2d
{
private :
    bool opened = false;
    Glpa::Image* ptBackGround;

public :
    ~ExampleScene2d() override;

    void openExample3d();
    
    void setup() override;

    void start() override;

    void update() override;
};

#endif EXAMPLE_SCENE_2D_H_
