#ifndef GLPA_RENDER_CU_H_
#define GLPA_RENDER_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SceneObject.h"

#include <unordered_map>
#include <Window.h>

namespace Glpa
{

class Render2d
{
private :

public :
    Render2d();
    ~Render2d();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*> objs,
        std::unordered_map<int, std::vector<std::string>> order,
        HDC dc, LPDWORD buf
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