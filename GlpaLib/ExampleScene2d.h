#ifndef EXAMPLE_SCENE_2D_H_
#define EXAMPLE_SCENE_2D_H_

#pragma warning(disable : 4996)

#include "Scene2d.h"

#include <string>
#include <Windows.h>

class ExampleScene2d : public Glpa::Scene2d
{
private :
    Glpa::Vec2d lastMouseLDownPos;

    bool opened = false;

    Glpa::Image* pBackground;
    bool isBackgroundMoving = false;
    Glpa::Vec2d lastBackgroundPos;

    Glpa::Image* pRect;
    bool isRectMoving = false;
    Glpa::Vec2d lastRectPos;

    Glpa::Image* pRect2;
    bool isRect2Moving = false;
    Glpa::Vec2d lastRect2Pos;

    bool anyImgMoving = false;

public :
    ~ExampleScene2d() override;

    void openExample3d();
    
    void setup() override;
    void start() override;
    void update() override;

    void imgMoveByMouse
    (
        std::string startMsg, std::string processMsg, std::string endMsg,
        Glpa::Image *target, Glpa::Vec2d& lastPos, bool &isMoving
    );

    void imgMoveStart(Glpa::Image *target, Glpa::Vec2d& lastPos, bool &isMoving);
    void imgMoving(Glpa::Image *target, Glpa::Vec2d& lastPos, bool &isMoving, Glpa::Vec2d &pos);
    void imgMoveStop(bool &isMoving);
};

#endif EXAMPLE_SCENE_2D_H_
