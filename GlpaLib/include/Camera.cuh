#ifndef GLPA_CAMERA_H_
#define GLPA_CAMERA_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Matrix.cuh"
#include "TriangleRatio.h"
#include "RangeRect.cuh"

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

    Glpa::GPU_VEC_2D nearScrSize;
    Glpa::GPU_VEC_2D farScrSize;
    Glpa::GPU_VEC_2D scrSize;

    Glpa::GPU_MAT_4X4 mtTransRot;
    Glpa::GPU_MAT_4X4 mtRot;

    Glpa::GPU_VIEW_VOLUME vv;

    __device__ __host__ GPU_BOOL isInside(Glpa::GPU_VEC_3D point)
    {
        Glpa::GPU_VECTOR_MG vecMgr;
        Glpa::GPU_VEC_2D vec[2] = 
        {
            Glpa::GPU_VEC_2D(point.x, point.z),
            Glpa::GPU_VEC_2D(point.y, point.z)
        };

        Glpa::GPU_VEC_2D axisVec(0, -1);

        float vecCos[2] = 
        {
            vecMgr.cos(vec[0], axisVec),
            vecMgr.cos(vec[1], axisVec)
        };

        GPU_BOOL isZIn = GPU_CO(point.z >= -farZ && point.z <= -nearZ, TRUE, FALSE);
        GPU_BOOL isXzIn = GPU_CO(vecCos[0] >= fovXzCos, TRUE, FALSE);
        GPU_BOOL isYzIn = GPU_CO(vecCos[1] >= fovYzCos, TRUE, FALSE);

        GPU_BOOL isIn = GPU_CO
        (
            isZIn == TRUE && isXzIn == TRUE && isYzIn == TRUE, 
            TRUE, FALSE
        );

        return isIn;
    }

    __device__ __host__ GPU_BOOL isInside(Glpa::GPU_RANGE_RECT& rangeRect)
    {
        Glpa::GPU_VECTOR_MG vecMgr;

        // By looking two-dimensionally, 
        // it is possible to determine whether an object is even partially within the camera's viewing angle.
        Glpa::GPU_VEC_2D cullingVecs[4] = {
            {rangeRect.origin.x, rangeRect.opposite.z},
            {rangeRect.opposite.x, rangeRect.opposite.z},
            {rangeRect.origin.y, rangeRect.opposite.z},
            {rangeRect.opposite.y, rangeRect.opposite.z}
        };

        Glpa::GPU_VEC_2D axisVec(0, -1);

        // Calculate the cosine of the angle between the culling vector and the axis vector.
        float vecCos[4] = {
            vecMgr.cos(cullingVecs[0], axisVec),
            vecMgr.cos(cullingVecs[1], axisVec),
            vecMgr.cos(cullingVecs[2], axisVec),
            vecMgr.cos(cullingVecs[3], axisVec)
        };

        GPU_BOOL isZIn = GPU_CO
        (
            rangeRect.origin.z >= -farZ && rangeRect.opposite.z <= -nearZ, 
            TRUE, FALSE
        );

        // Whether the object is within the camera's viewing angle when viewed along the Xz axis.
        GPU_BOOL isXzIn = GPU_CO
        (
            (rangeRect.origin.x >= 0 && vecCos[0] >= fovXzCos) || 
            (rangeRect.opposite.x <= 0 && vecCos[1] >= fovXzCos) ||
            (rangeRect.origin.x <= 0 && rangeRect.opposite.x >= 0),
            TRUE, FALSE
        );

        // Whether the object is within the camera's viewing angle when viewed along the Yz axis.
        GPU_BOOL isYzIn = GPU_CO
        (
            (rangeRect.origin.y >= 0 && vecCos[2] >= fovYzCos) || 
            (rangeRect.opposite.y <= 0 && vecCos[3] >= fovYzCos) ||
            (rangeRect.origin.y <= 0 && rangeRect.opposite.y >= 0),
            TRUE, FALSE
        );

        GPU_BOOL isIn = GPU_CO
        (
            isZIn == TRUE && isXzIn == TRUE && isYzIn == TRUE, 
            TRUE, FALSE
        );

        return isIn;
    }

    __device__ __host__ Glpa::GPU_VEC_3D getScrPos(int i, Glpa::GPU_VEC_3D point)
    {
        float scrX = -nearZ * point.x / point.z + nearScrSize.x / 2;
        float scrY = -nearZ * point.y / point.z + nearScrSize.y / 2;

        int scrPixelX = (scrX / nearScrSize.x) * scrSize.x;
        int scrPixelY = scrSize.y - (scrY / nearScrSize.y) * scrSize.y;
        Glpa::GPU_VEC_3D scrPos = {(float)scrPixelX, (float)scrPixelY, point.z};

        return scrPos;
    }

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
    Glpa::Vec2d scrSize;

public :
    Camera
    (
        std::string argName, Glpa::Vec3d defPos, Glpa::Vec3d defRotate, 
        float defFov, Glpa::Vec2d defAspectRatio, float defNearZ, float defFarZ, Glpa::Vec2d defScrSize
    );
    ~Camera();

    std::string getName() const {return name;}

    Glpa::GPU_CAMERA getData();

};

class CAMERA_FACTORY
{
public :
    bool malloced = false;

    CAMERA_FACTORY()
    {
        malloced = false;
    }

    void dFree(Glpa::GPU_CAMERA*& dCamera);
    void dMalloc(Glpa::GPU_CAMERA*& dCamera, Glpa::GPU_CAMERA& hCamera);

    void deviceToHost(Glpa::GPU_CAMERA*& dCamera, Glpa::GPU_CAMERA& hCamera);
};


}; // namespace Glpa



#endif GLPA_CAMERA_H_