#include "ExampleScene3d.h"

ExampleScene3d::~ExampleScene3d()
{
}

void ExampleScene3d::setup()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_EXAMPLE, "Scene[" + name + "]");

    Glpa::Material *ptMaterial = new Glpa::Material("mt_cube", "resource/Assets/Mtls/Cube_BaseColor.png");
    AddMaterial(ptMaterial);

    Glpa::StationaryObject* ptCube = new Glpa::StationaryObject("Cube", "resource/Assets/Objs/Cube.obj", Glpa::Vec3d(0.0, 0.0, 0.0));
    ptCube->SetMaterial(ptMaterial);
    AddSceneObject(ptCube);

    Glpa::Camera* ptCam = new Glpa::Camera
    (
        "Main Camera", Glpa::Vec3d(0.0, 0.0, 0.0), Glpa::Vec3d(0.0, 0.0, 0.0),
        90, Glpa::Vec2d(16, 9), 0.1, 1000
    );
    AddCamera(ptCam);
    SetCamera(ptCam);
    
}

void ExampleScene3d::start()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_EXAMPLE, "ExampleScene3d::start()");
}
