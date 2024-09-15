#include "ExampleScene3d.h"

ExampleScene3d::~ExampleScene3d()
{
}

void ExampleScene3d::setup()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_EXAMPLE, "Scene[" + name + "]");

    Glpa::Material *ptMtCube = new Glpa::Material("mt_cube", "resource/Assets/Mtls/cube/cube_baseColor.png");
    AddMaterial(ptMtCube);

    Glpa::StationaryObject* ptCube = new Glpa::StationaryObject("Cube", "resource/Assets/Objs/Cube.obj", Glpa::Vec3d(0.0, 0.0, 0.0));
    ptCube->SetMaterial(ptMtCube);
    AddSceneObject(ptCube);

    Glpa::Material* ptMtFloor = new Glpa::Material("mt_floor", "resource/Assets/Mtls/floor/floor_baseColor.png");
    AddMaterial(ptMtFloor);

    Glpa::StationaryObject* ptFloor = new Glpa::StationaryObject("Floor", "resource/Assets/Objs/Floor.obj", Glpa::Vec3d(0.0, 0.0, 0.0));
    ptFloor->SetMaterial(ptMtFloor);
    AddSceneObject(ptFloor);

    Glpa::Camera* ptCam = new Glpa::Camera
    (
        "Main Camera", Glpa::Vec3d(0.0, 0.0, 0.0), Glpa::Vec3d(0.0, 0.0, 0.0),
        80, Glpa::Vec2d(16, 9), 0.1, 10000, {1920, 1080}
    );
    AddCamera(ptCam);
    SetCamera(ptCam);
    
}

void ExampleScene3d::start()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_EXAMPLE, "ExampleScene3d::start()");
}
