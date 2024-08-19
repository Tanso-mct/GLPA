#ifndef GLPA_CAMERA_H_
#define GLPA_CAMERA_H_

#include "ViewVolume.h"

namespace Glpa
{

typedef struct _CAMERA
{
    float pos[3];
    float rotate[3];
    float fov;
    float aspectRatio[2];
    float nearZ;
    float farZ;
} CAMERA;

class Camera
{
private :
    std::string name;
    Glpa::Vec3d pos;
    Glpa::Vec3d rotate;

    float fov = 0.0f;
    Glpa::Vec2d aspectRatio;
    float nearZ = 0.0f;
    float farZ = 0.0f;

    Glpa::ViewVolume* vv = nullptr;

public :
    Camera
    (
        std::string argName, Glpa::Vec3d defPos, Glpa::Vec3d defRotate, 
        float defFov, Glpa::Vec2d defAspectRatio, float defNearZ, float defFarZ
    );
    ~Camera();

    std::string getName() const {return name;}

    Glpa::CAMERA getData();

};



}



#endif GLPA_CAMERA_H_