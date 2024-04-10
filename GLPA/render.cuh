#ifndef RENDER_CUH_
#define RENDER_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <unordered_map>
#include <string>
#include <cmath>

#include "object.h"
#include "camera.cuh"
#include "cg.h"

__global__ void glpaGpuPreparePoly(
    int obj_size,
    double* obj_world_vs
);


class Render{
public :
    void prepareObjs(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam
    );

    void render(
        std::unordered_map<std::wstring, Object> source_objects,
        Camera cam,
        LPDWORD buffer
    );

private :

};


#endif RENDER_CUH_