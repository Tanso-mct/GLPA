#include "Scene.h"

Glpa::Scene::Scene()
{
    awake();
}

Glpa::Scene::~Scene()
{
    destroy();
}

void Glpa::Scene::getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam)
{

}

void Glpa::Scene::getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam)
{

}

void Glpa::Scene::getMouse(UINT msg, WPARAM wParam, LPARAM lParam)
{

}

void Glpa::Scene::getNowKeyMsg()
{

}

void Glpa::Scene::getNowMouseMsg()
{

}

void Glpa::Scene::AddSceneObject(Glpa::SceneObject *ptObj)
{

}

void Glpa::Scene::DeleteSceneObject(Glpa::SceneObject *ptObj)
{
    
}
