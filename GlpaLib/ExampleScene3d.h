#ifndef EXAMPLE_SCENE_3D_H_
#define EXAMPLE_SCENE_3D_H_

#include "Scene3d.h"

class ExampleScene3d : public Glpa::Scene3d
{
private :

public :
    ~ExampleScene3d() override;
    
    void setup() override;

    void start() override;
};

#endif EXAMPLE_SCENE_3D_H_
