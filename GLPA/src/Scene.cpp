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
    objs.emplace(ptObj->getName(), ptObj);
}

void Glpa::Scene::DeleteSceneObject(Glpa::SceneObject *ptObj)
{
    ptObj->release();

    objs.erase(ptObj->getName());
    delete ptObj;
}
