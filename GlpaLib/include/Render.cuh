#ifndef GLPA_RENDER_CU_H_
#define GLPA_RENDER_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SceneObject.h"

#include "Image.h"
#include "Color.h"

#include <unordered_map>
#include <map>
#include <Window.h>

namespace Glpa
{

__global__ void Gpu2dDraw
(
    int* imgPosX,
    int* imgPosY,
    int* imgWidth,
    int* imgHeight,
    LPDWORD* imgData,
    int imgAmount,
    LPDWORD buf,
    int bufWidth,
    int bufHeight,
    int bufDpi,
    DWORD background
);

__global__ void Gpu2dDrawBackground
(
    LPDWORD buf,
    int bufWidth,
    int bufHeight,
    int bufDpi,
    DWORD background
);

class Render2d
{
private :

public :
    Render2d();
    ~Render2d();

    void setBackground(std::string color, DWORD& bg);

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*> objs,
        std::map<int, std::vector<std::string>> drawOrder,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
    );
};

class Render3d
{
private :

public :
    Render3d();
    ~Render3d();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*> objs, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
    );
};

}

#endif GLPA_RENDER_CU_H_