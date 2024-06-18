#ifndef GLPA_SCENE_H_
#define GLPA_SCENE_H_

#include <string>
#include <Windows.h>
#include <unordered_map>
#include <cctype>

#include "Image.h"
#include "Constant.h"
#include "Render.cuh"

#include <d2d1.h>
#pragma comment(lib, "d2d1")

namespace Glpa
{

class Scene
{
private :
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

protected :
    std::string name;
    std::unordered_map<std::string, Glpa::SceneObject*> objs;

    std::string backgroundColor = Glpa::COLOR_BLACK;

public :
    Scene();
    virtual ~Scene();

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    /// @brief Handle keydown messages in a window procedure function.
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

    std::string GetBgColor() const {return backgroundColor;}
    void SetBgColor(std::string str) {backgroundColor = str;}

    /// @brief Add scene objects.
    virtual void setup() = 0;

    virtual void start(){};
    virtual void update(){};

    virtual void awake(){};
    virtual void destroy(){};

    virtual void load() = 0;
    virtual void release() = 0;

    virtual void rendering(ID2D1HwndRenderTarget*& pRenderTarget, ID2D1Bitmap*& pBitMap, LPDWORD buf, int& bufWidth, int& bufHeight, int& bufDpi) = 0;

};

}

#endif GLPA_SCENE_H_
