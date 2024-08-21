#ifndef GLPA_RANGE_RECT_H_
#define GLPA_RANGE_RECT_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <Windows.h>

#include "Vector.cuh"

namespace Glpa
{

typedef struct _GPU_RANGE_RECT
{
    GPU_BOOL isEmpty = FALSE;
    GPU_VEC_3D origin;
    GPU_VEC_3D opposite;
    GPU_VEC_3D wv[8];

    __device__ void addRangeV(GPU_VEC_3D wv)
    {
        GPU_IF(!isEmpty, b2)
        {
            origin = wv;
            opposite = wv;
            isEmpty = TRUE;
            return;
        }
        GPU_IF(isEmpty, b2)
        {
            origin.x = GPU_CO(wv.x < origin.x, wv.x, origin.x);
            origin.y = GPU_CO(wv.y < origin.y, wv.y, origin.y);
            origin.z = GPU_CO(wv.z > origin.z, wv.z, origin.z);

            opposite.x = GPU_CO(wv.x > opposite.x, wv.x, opposite.x);
            opposite.y = GPU_CO(wv.y > opposite.y, wv.y, opposite.y);
            opposite.z = GPU_CO(wv.z < opposite.z, wv.z, opposite.z);
        }

        isEmpty = TRUE;
    }

    __device__ void setWvs()
    {
        wv[0] = GPU_VEC_3D(origin.x, opposite.y, origin.z);
        wv[1] = GPU_VEC_3D(opposite.x, opposite.y, origin.z);
        wv[2] = GPU_VEC_3D(opposite.x, origin.y, origin.z);
        wv[3] = GPU_VEC_3D(origin.x, origin.y, origin.z);
        wv[4] = GPU_VEC_3D(origin.x, opposite.y, opposite.z);
        wv[5] = GPU_VEC_3D(opposite.x, opposite.y, opposite.z);
        wv[6] = GPU_VEC_3D(opposite.x, origin.y, opposite.z);
        wv[7] = GPU_VEC_3D(origin.x, origin.y, opposite.z);
    }
} GPU_RANGE_RECT;

class RangeRect
{
private :
    bool isEmpty = false;
    Glpa::Vec3d origin;
    Glpa::Vec3d opposite;
    std::vector<Glpa::Vec3d> wvs;

public :
    RangeRect();
    ~RangeRect();

    Glpa::Vec3d getOrigin() const {return origin;}
    Glpa::Vec3d getOpposite() const {return opposite;}

    void addRangeV(Glpa::Vec3d* wv);

    void setWvs();
    Glpa::GPU_RANGE_RECT getData();
};


}

#endif GLPA_RANGE_RECT_H_