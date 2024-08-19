#include "Camera.h"
#include "GlpaLog.h"

Glpa::Camera::Camera(std::string argName, Glpa::Vec3d defPos, Glpa::Vec3d defRotate, float defFov, Glpa::Vec2d defAspectRatio, float defNearZ, float defFarZ)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    pos = defPos;
    rotate = defRotate;
    fov = defFov;
    aspectRatio = defAspectRatio;
    nearZ = defNearZ;
    farZ = defFarZ;
}

Glpa::Camera::~Camera()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

Glpa::CAMERA Glpa::Camera::getData()
{
    Glpa::CAMERA camera;

    camera.pos[X] = pos.x;
    camera.pos[Y] = pos.y;
    camera.pos[Z] = pos.z;

    camera.rotate[X] = rotate.x;
    camera.rotate[Y] = rotate.y;
    camera.rotate[Z] = rotate.z;

    camera.fov = fov;

    camera.aspectRatio[X] = aspectRatio.x;
    camera.aspectRatio[Y] = aspectRatio.y;

    camera.nearZ = nearZ;
    camera.farZ = farZ;

    return camera;
}

