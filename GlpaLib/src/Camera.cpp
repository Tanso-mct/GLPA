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

    // Get screen size
    nearScrSize.x = fabs(tan(Glpa::RtoD(fov / 2)) * -nearZ) * 2;
    nearScrSize.y = fabs(nearScrSize.x * aspectRatio.y / aspectRatio.x);

    farScrSize.x = fabs(tan(Glpa::RtoD(fov / 2)) * -farZ) * 2;
    farScrSize.y = fabs(farScrSize.x * aspectRatio.y / aspectRatio.x);

    fovXzCos = cos(Glpa::RtoD(fov / 2));
    fovYzCos = fabs(-nearZ / sqrt(-nearZ*-nearZ + (nearScrSize.y/2) * (nearScrSize.y/2)));
}

Glpa::Camera::~Camera()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

Glpa::GPU_CAMERA Glpa::Camera::getData()
{
    Glpa::GPU_CAMERA camera;

    camera.pos.x = pos.x;
    camera.pos.y = pos.y;
    camera.pos.z = pos.z;

    camera.rotate.x = rotate.x;
    camera.rotate.y = rotate.y;
    camera.rotate.z = rotate.z;

    camera.fov = fov;
    camera.fovXzCos = fovXzCos;
    camera.fovYzCos = fovYzCos;

    camera.aspectRatio.x = aspectRatio.x;
    camera.aspectRatio.y = aspectRatio.y;

    camera.nearZ = nearZ;
    camera.farZ = farZ;

    float mt4x4[16] =
    {
        std::cos(Glpa::RtoD(rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)), 
        std::cos(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) - std::sin(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)), 
        -pos.x,

        std::sin(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.y)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * std::cos(Glpa::RtoD(-rotate.x)),
        std::sin(Glpa::RtoD(-rotate.z)) * std::sin(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)) + std::cos(Glpa::RtoD(-rotate.z)) * -std::sin(Glpa::RtoD(-rotate.x)),
        -pos.y,

        -std::sin(Glpa::RtoD(-rotate.y)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::sin(Glpa::RtoD(-rotate.x)),
        std::cos(Glpa::RtoD(-rotate.y)) * std::cos(Glpa::RtoD(-rotate.x)),
        -pos.z,

        0, 0, 0, 1
    };

    camera.mtTransRot.set(mt4x4);
    return camera;
}

