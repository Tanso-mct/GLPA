#ifndef GLPA_SCENE_H_
#define GLPA_SCENE_H_

#include <string>
#include <Windows.h>
#include <unordered_map>
#include <cctype>

#include "Image.h"
#include "Constant.h"

namespace Glpa
{

class Scene
{
protected :
    std::string name;
    std::unordered_map<std::string, Glpa::SceneObject*> objs;

    bool shiftToggle = false;
    bool ctrlToggle = false;
    bool altToggle = false;

    std::string keyMsg;
    std::string keyDownMsg;
    std::string keyUpMsg;

    std::string mouseMsg;
    Glpa::Vec2d mousePos;

    Glpa::Vec2d moouseRDownPos;
    Glpa::Vec2d moouseRDbClickPos;
    Glpa::Vec2d moouseRUpPos;

    Glpa::Vec2d moouseLDownPos;
    Glpa::Vec2d moouseLDbClickPos;
    Glpa::Vec2d moouseLUpPos;

    Glpa::Vec2d moouseMDownPos;
    Glpa::Vec2d moouseMUpPos;
    short wheelMoveAmount = 0;

public :
    Scene();
    virtual ~Scene();

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}
    
    void getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam);
    void getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam);
    void getMouse(UINT msg, WPARAM wParam, LPARAM lParam, int dpi);

    std::string GetNowKeyMsg();
    std::string GetNowKeyDownMsg();
    std::string GetNowKeyUpMsg();

    void updateKeyMsg();
    void updateMouseMsg();

    void GetNowMouseMsg();

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
