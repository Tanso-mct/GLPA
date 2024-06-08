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

    Glpa::Vec2d mouseRDownPos;
    Glpa::Vec2d mouseRDbClickPos;
    Glpa::Vec2d mouseRUpPos;

    Glpa::Vec2d mouseLDownPos;
    Glpa::Vec2d mouseLDbClickPos;
    Glpa::Vec2d mouseLUpPos;

    Glpa::Vec2d mouseMDownPos;
    Glpa::Vec2d mouseMUpPos;
    short wheelMoveAmount = 0;

public :
    Scene();
    virtual ~Scene();

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}
    
    void getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam);
    void getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam);
    void getMouse(UINT msg, WPARAM wParam, LPARAM lParam, int dpi);

    bool IsShiftToggle() const {return shiftToggle;}
    bool IsCtrlToggle() const {return ctrlToggle;}
    bool IsAltToggle() const {return altToggle;}

    std::string GetNowKeyMsg();
    bool GetNowKeyMsg(std::string argMsg);

    std::string GetNowKeyDownMsg();
    bool GetNowKeyDownMsg(std::string argMsg);

    std::string GetNowKeyUpMsg();
    bool GetNowKeyUpMsg(std::string argMsg);

    void updateKeyMsg();

    std::string GetNowMouseMsg();
    bool GetNowMouseMsg(std::string argMsg);
    bool GetNowMouseMsg(std::string argMsg, Glpa::Vec2d &target);
    bool GetNowMouseMsg(std::string argMsg, int &amount);

    void updateMouseMsg();

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
