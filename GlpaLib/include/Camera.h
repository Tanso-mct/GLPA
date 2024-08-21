#ifndef GLPA_CAMERA_H_
#define GLPA_CAMERA_H_

#include "Matrix.cuh"
#include "TriangleRatio.h"

#include <cmath>
#include <string>

namespace Glpa
{

typedef struct _GPU_CAMERA
{
    Glpa::GPU_VEC_3D pos;
    Glpa::GPU_VEC_3D rotate;

    float fov;
    float fovXzCos;
    float fovYzCos;

    Glpa::GPU_VEC_2D aspectRatio;
    float nearZ;
    float farZ;

    Glpa::GPU_MAT_4X4 mtTransRot;
} GPU_CAMERA;

class Camera
{
private :
    std::string name;
    Glpa::Vec3d pos;
    Glpa::Vec3d rotate;

    float fov = 0.0f;
    float fovXzCos = 0.0f;
    float fovYzCos = 0.0f;

    Glpa::Vec2d aspectRatio;
    float nearZ = 0.0f;
    float farZ = 0.0f;

    Glpa::Vec2d nearScrSize;
    Glpa::Vec2d farScrSize;

public :
    Camera
    (
        std::string argName, Glpa::Vec3d defPos, Glpa::Vec3d defRotate, 
        float defFov, Glpa::Vec2d defAspectRatio, float defNearZ, float defFarZ
    );
    ~Camera();

    std::string getName() const {return name;}

    Glpa::GPU_CAMERA getData();

};



}



#endif GLPA_CAMERA_H_