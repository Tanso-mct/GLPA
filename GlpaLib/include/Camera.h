#ifndef GLPA_CAMERA_H_
#define GLPA_CAMERA_H_

#include "Matrix.cuh"
#include "TriangleRatio.h"

#include <cmath>
#include <string>

namespace Glpa
{

typedef struct _GPU_VIEW_VOLUME
{
    Glpa::GPU_VEC_3D wv[8];

    Glpa::GPU_VEC_2D xzV[4];
    Glpa::GPU_VEC_2D yzV[4];

    Glpa::GPU_LINE_3D line[12];
    Glpa::GPU_FACE_3D face[6];
} GPU_VIEW_VOLUME;

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

    Glpa::GPU_VIEW_VOLUME vv;
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