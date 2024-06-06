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
    switch (wParam)
    {
        case VK_SHIFT:
            shiftToggle = true;
            break;

        case VK_CONTROL:
            ctrlToggle = true;
            break;

        case VK_MENU:
            altToggle = true;
            break;

        case '0':
            keyMsg = "0";
            break;

        case '1':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '2':
            keyMsg = (shiftToggle) ? "\"" : "2";
            break;

        case '3':
            keyMsg = (shiftToggle) ? "#" : "3";
            break;

        case '4':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '5':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '6':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '7':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '8':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case '9':
            keyMsg = (shiftToggle) ? "!" : "0";
            break;

        case 'A':
        case 'B':
        case 'C':
        case 'D':
        case 'E':
        case 'F':
        case 'G':
        case 'H':
        case 'I':
        case 'J':
        case 'K':
        case 'L':
        case 'M':
        case 'N':
        case 'O':
        case 'P':
        case 'Q':
        case 'R':
        case 'S':
        case 'T':
        case 'U':
        case 'V':
        case 'W':
        case 'X':
        case 'Y':
        case 'Z':
            if (shiftToggle) {
                keyMsg = std::string(1, static_cast<char>(wParam));
            }
            else
            {
                keyMsg = std::string(1, std::tolower(static_cast<char>(wParam)));
            }
            break;

    }
}

void Glpa::Scene::getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam)
{

}

void Glpa::Scene::getMouse(UINT msg, WPARAM wParam, LPARAM lParam)
{

}

void Glpa::Scene::GetNowKeyMsg()
{

}

void Glpa::Scene::GetNowMouseMsg()
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

void Glpa::Scene::load()
{
    for (auto& obj : objs)
    {
        obj.second->load();
    }
}

void Glpa::Scene::release()
{
    for (auto& obj : objs)
    {
        obj.second->release();
    }
}
