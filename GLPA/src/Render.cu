#include "Render.cuh"

Glpa::Render2d::Render2d()
{
}

Glpa::Render2d::~Render2d()
{
}

void Glpa::Render2d::run
(
    std::unordered_map<std::string, Glpa::SceneObject*> objs,
    std::unordered_map<int, std::vector<std::string>> order,
    HDC dc, LPDWORD buf
){

}

Glpa::Render3d::Render3d()
{
}

Glpa::Render3d::~Render3d()
{
}

void Glpa::Render3d::run(std::unordered_map<std::string, Glpa::SceneObject*> objs, HDC dc, LPDWORD buf)
{
    
}
