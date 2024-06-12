#ifndef GLPA_RENDER_CU_H_
#define GLPA_RENDER_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SceneObject.h"

#include "Image.h"

#include <unordered_map>
#include <map>
#include <Window.h>

namespace Glpa
{

__global__ void GpuDrawBufBackground(
    double* world_vs,
    double* near_z,
    double* far_z,
    double* near_screen_size,
    double* screen_pixel_size,
    double* result_vs,
    int world_vs_amount
);

class Render2d
{
private :

public :
    Render2d();
    ~Render2d();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*> objs,
        std::map<int, std::vector<std::string>> drawOrder,
        HDC dc, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
    );
};

class Render3d
{
private :

public :
    Render3d();
    ~Render3d();

    void run(std::unordered_map<std::string, Glpa::SceneObject*> objs, HDC dc, LPDWORD buf);
};

}

#endif GLPA_RENDER_CU_H_