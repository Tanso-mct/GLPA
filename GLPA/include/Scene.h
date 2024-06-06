#ifndef GLPA_SCENE_H_
#define GLPA_SCENE_H_

#include <string>
#include <Windows.h>
#include <unordered_map>

#include "Image.h"

namespace Glpa
{

class Scene
{
protected :
    std::string name;
    std::unordered_map<std::string, Glpa::SceneObject*> objs;

    std::string keyMsg;
    bool keyMsgUpdated = false;

    std::string mouseMsg;
    bool mouseMsgUpdated = false;

public :
    Scene();
    virtual ~Scene();

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}
    
    void getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam);
    void getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam);
    void getMouse(UINT msg, WPARAM wParam, LPARAM lParam);

    void getNowKeyMsg();
    void getNowMouseMsg();

    void AddSceneObject(Glpa::SceneObject* ptObj);
    void DeleteSceneObject(Glpa::SceneObject* ptObj);

    /// @brief Add scene objects.
    virtual void setup() = 0;

    virtual void start(){};
    virtual void update(){};

    virtual void awake(){};
    virtual void destroy(){};

    void load();
    void release();

};

}

#endif GLPA_SCENE_H_
