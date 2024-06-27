#include "ExampleScene2d.h"

ExampleScene2d::~ExampleScene2d()
{
}

void ExampleScene2d::setup()
{
    std::string folderPath = "resource/Assets/Images/";
    pBackground = new Glpa::Image("back_ground",folderPath + "console_back_ground.png", Glpa::Vec2d(0, 0));
    AddSceneObject(pBackground);

    pRect = new Glpa::Image("rect",folderPath + "rect.png", Glpa::Vec2d(0, 0));
    pRect->SetDrawOrder(1);
    AddSceneObject(pRect);

    pRect2 = new Glpa::Image("rect2",folderPath + "rect2.png", Glpa::Vec2d(0, 0));
    pRect2->SetDrawOrder(2);
    AddSceneObject(pRect2);

    SetBgColor(Glpa::COLOR_GREEN);
}

void ExampleScene2d::start()
{
    
}

void ExampleScene2d::update()
{
    // if (!GetNowKeyDownMsg().empty())
    // {
    //     OutputDebugStringA("Key Down ");
    //     OutputDebugStringA(GetNowKeyDownMsg().c_str());
    //     OutputDebugStringA("\n");
    // }

    // if (!GetNowKeyMsg().empty())
    // {
    //     OutputDebugStringA("Key ");
    //     OutputDebugStringA(GetNowKeyMsg().c_str());
    //     OutputDebugStringA("\n");
    // }

    // if (!GetNowKeyUpMsg().empty())
    // {
    //     OutputDebugStringA("Key Up ");
    //     OutputDebugStringA(GetNowKeyUpMsg().c_str());
    //     OutputDebugStringA("\n");
    // }

    // if (GetNowKeyDownMsg("g")) OutputDebugStringA("Key Down g\n");
    // else if (GetNowKeyDownMsg("G")) OutputDebugStringA("Key Down G\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_LSHIFT)) OutputDebugStringA("Key Down Left Shift\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_LCTRL)) OutputDebugStringA("Key Down Left Ctrl\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_LALT)) OutputDebugStringA("Key Down Left Alt\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_WIN)) OutputDebugStringA("Key Down Windows\n");

    // else if (GetNowKeyDownMsg(Glpa::CHAR_RSHIFT)) OutputDebugStringA("Key Down Right Shift\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_RCTRL)) OutputDebugStringA("Key Down Right Ctrl\n");
    // else if (GetNowKeyDownMsg(Glpa::CHAR_RALT)) OutputDebugStringA("Key Down Right Alt\n");

    // else if (GetNowKeyDownMsg(Glpa::CHAR_F1)) OutputDebugStringA("Key Down F1\n");

    // else if (GetNowKeyDownMsg("1")) OutputDebugStringA("Key Down 1\n");
    // else if (GetNowKeyDownMsg("#")) OutputDebugStringA("Key Down #\n");

    // if (!GetNowMouseMsg().empty() && GetNowMouseMsg() != Glpa::CHAR_MOUSE_MOVE)
    // {
    //     OutputDebugStringA("Mouse ");
    //     OutputDebugStringA(GetNowMouseMsg().c_str());
    //     OutputDebugStringA("\n");
    // }

    moveImgByLBtn(pRect, lastRectPos, isRectMoving);
    moveImgByRBtn(pBackground, lastBackgroundPos, isBackgroundMoving);
    moveImgByMBtn(pRect2, lastRect2Pos, isRect2Moving);

    if (GetNowKeyDownMsg("g")) 
    {
        if (!opened)
        {
            OutputDebugStringA("Open example 3d\n");
            openExample3d();
        }
    }
}

void ExampleScene2d::moveImgByLBtn(Glpa::Image *target, Glpa::Vec2d& lastPos, bool &isMoving)
{
    Glpa::Vec2d pos;
    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_LBTN_DOWN, pos) && !isMoving)
    {
        lastMouseLDownPos = pos;
        lastPos = target->GetPos();
        isMoving = true;
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_MOVE, pos) && isMoving)
    {
        Glpa::Vec2d moveCoord(lastPos.x + pos.x - lastMouseLDownPos.x, lastPos.y + pos.y - lastMouseLDownPos.y);
        editPos(target, moveCoord);
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_LBTN_UP) && isMoving)
    {
        isMoving = false;
    }
}

void ExampleScene2d::moveImgByRBtn(Glpa::Image *target, Glpa::Vec2d& lastPos, bool &isMoving)
{
    Glpa::Vec2d pos;
    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_RBTN_DOWN, pos) && !isMoving)
    {
        lastMouseLDownPos = pos;
        lastPos = target->GetPos();
        isMoving = true;
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_MOVE, pos) && isMoving)
    {
        Glpa::Vec2d moveCoord(lastPos.x + pos.x - lastMouseLDownPos.x, lastPos.y + pos.y - lastMouseLDownPos.y);
        editPos(target, moveCoord);
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_RBTN_UP) && isMoving)
    {
        isMoving = false;
    }
}

void ExampleScene2d::moveImgByMBtn(Glpa::Image *target, Glpa::Vec2d &lastPos, bool &isMoving)
{
    Glpa::Vec2d pos;
    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_MBTN_DOWN, pos) && !isMoving)
    {
        lastMouseLDownPos = pos;
        lastPos = target->GetPos();
        isMoving = true;
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_MOVE, pos) && isMoving)
    {
        Glpa::Vec2d moveCoord(lastPos.x + pos.x - lastMouseLDownPos.x, lastPos.y + pos.y - lastMouseLDownPos.y);
        editPos(target, moveCoord);
    }

    if (GetNowMouseMsg(Glpa::CHAR_MOUSE_MBTN_UP) && isMoving)
    {
        isMoving = false;
    }
}
