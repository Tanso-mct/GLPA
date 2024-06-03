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
    ~Scene();

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}
    
    void getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam);
    void getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam);
    void getMouse(UINT msg, WPARAM wParam, LPARAM lParam);

    void getNowKeyMsg();
    void getNowMouseMsg();

    void addSceneObject(Glpa::SceneObject* ptObj);
    void deleteSceneObject(Glpa::SceneObject* ptObj);

    virtual void setup() = 0;

    virtual void start(){};
    virtual void update(){};

    virtual void awake(){};
    virtual void destroy(){};

    virtual void load() = 0;
    virtual void release() = 0;

};

}

#endif GLPA_SCENE_H_
