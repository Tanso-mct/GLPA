#include "ExampleScene2d.h"

ExampleScene2d::~ExampleScene2d()
{
}

void ExampleScene2d::setup()
{
    ptBackGround = new Glpa::Image("back_ground","resource/Assets/Images/console_back_ground.png", Glpa::Vec2d(0, 0));

    AddSceneObject(ptBackGround);
}

void ExampleScene2d::update()
{
    if (!GetNowKeyDownMsg().empty())
    {
        OutputDebugStringA("Key Down ");
        OutputDebugStringA(GetNowKeyDownMsg().c_str());
        OutputDebugStringA("\n");
    }

    if (!GetNowKeyMsg().empty())
    {
        OutputDebugStringA("Key ");
        OutputDebugStringA(GetNowKeyMsg().c_str());
        OutputDebugStringA("\n");
    }

    if (!GetNowKeyUpMsg().empty())
    {
        OutputDebugStringA("Key Up ");
        OutputDebugStringA(GetNowKeyUpMsg().c_str());
        OutputDebugStringA("\n");
    }

    if (GetNowKeyDownMsg("g")) OutputDebugStringA("Key Down g\n");
    else if (GetNowKeyDownMsg("G")) OutputDebugStringA("Key Down G\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_LSHIFT)) OutputDebugStringA("Key Down Left Shift\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_LCTRL)) OutputDebugStringA("Key Down Left Ctrl\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_LALT)) OutputDebugStringA("Key Down Left Alt\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_WIN)) OutputDebugStringA("Key Down Windows\n");

    else if (GetNowKeyDownMsg(Glpa::CHAR_RSHIFT)) OutputDebugStringA("Key Down Right Shift\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_RCTRL)) OutputDebugStringA("Key Down Right Ctrl\n");
    else if (GetNowKeyDownMsg(Glpa::CHAR_RALT)) OutputDebugStringA("Key Down Right Alt\n");

    else if (GetNowKeyDownMsg(Glpa::CHAR_F1)) OutputDebugStringA("Key Down F1\n");

    else if (GetNowKeyDownMsg("1")) OutputDebugStringA("Key Down 1\n");
    else if (GetNowKeyDownMsg("#")) OutputDebugStringA("Key Down #\n");

}
