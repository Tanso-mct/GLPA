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
        case VK_SPACE:
            keyMsg = Glpa::CHAR_SPACE;
            keyDownMsg = Glpa::CHAR_SPACE;
            break;

        case VK_ESCAPE:
            keyMsg = Glpa::CHAR_ESCAPE;
            keyDownMsg = Glpa::CHAR_ESCAPE;
            break;

        case VK_TAB:
            keyMsg = Glpa::CHAR_TAB;
            keyDownMsg = Glpa::CHAR_TAB;
            break;

        case VK_BACK:
            keyMsg = Glpa::CHAR_BACKSPACE;
            keyDownMsg = Glpa::CHAR_BACKSPACE;
            break;

        case VK_LSHIFT:
            shiftToggle = true;
            keyMsg = Glpa::CHAR_LSHIFT;
            keyDownMsg = Glpa::CHAR_LSHIFT;
            break;

        case VK_LCONTROL:
            ctrlToggle = true;
            keyMsg = Glpa::CHAR_LCTRL;
            keyDownMsg = Glpa::CHAR_LCTRL;
            break;

        case VK_LMENU:
            altToggle = true;
            keyMsg = Glpa::CHAR_LALT;
            keyDownMsg = Glpa::CHAR_LALT;
            break;

        case VK_LWIN:
            keyMsg = Glpa::CHAR_LWIN;
            keyDownMsg = Glpa::CHAR_LWIN;
            break;

        case VK_RSHIFT:
            shiftToggle = true;
            keyMsg = Glpa::CHAR_RSHIFT;
            keyDownMsg = Glpa::CHAR_RSHIFT;
            break;

        case VK_RCONTROL:
            ctrlToggle = true;
            keyMsg = Glpa::CHAR_RCTRL;
            keyDownMsg = Glpa::CHAR_RCTRL;
            break;

        case VK_RMENU:
            altToggle = true;
            keyMsg = Glpa::CHAR_RALT;
            keyDownMsg = Glpa::CHAR_RALT;
            break;

        case VK_RWIN:
            keyMsg = Glpa::CHAR_RWIN;
            keyDownMsg = Glpa::CHAR_RWIN;
            break;

        case VK_F1:
            keyMsg = Glpa::CHAR_F1;
            keyDownMsg = Glpa::CHAR_F1;
            break;

        case VK_F2:
            keyMsg = Glpa::CHAR_F2;
            keyDownMsg = Glpa::CHAR_F2;
            break;

        case VK_F3:
            keyMsg = Glpa::CHAR_F3;
            keyDownMsg = Glpa::CHAR_F3;
            break;

        case VK_F4:
            keyMsg = Glpa::CHAR_F4;
            keyDownMsg = Glpa::CHAR_F4;
            break;

        case VK_F5:
            keyMsg = Glpa::CHAR_F5;
            keyDownMsg = Glpa::CHAR_F5;
            break;

        case VK_F6:
            keyMsg = Glpa::CHAR_F6;
            keyDownMsg = Glpa::CHAR_F6;
            break;

        case VK_F7:
            keyMsg = Glpa::CHAR_F7;
            keyDownMsg = Glpa::CHAR_F7;
            break;

        case VK_F8:
            keyMsg = Glpa::CHAR_F8;
            keyDownMsg = Glpa::CHAR_F8;
            break;

        case VK_F9:
            keyMsg = Glpa::CHAR_F9;
            keyDownMsg = Glpa::CHAR_F9;
            break;

        case VK_F10:
            keyMsg = Glpa::CHAR_F10;
            keyDownMsg = Glpa::CHAR_F10;
            break;

        case VK_F11:
            keyMsg = Glpa::CHAR_F11;
            keyDownMsg = Glpa::CHAR_F11;
            break;

        case VK_F12:
            keyMsg = Glpa::CHAR_F12;
            keyDownMsg = Glpa::CHAR_F12;
            break;

        case '0':
            keyMsg = "0";
            keyDownMsg = "0";
            break;

        case '1':
            keyMsg = (shiftToggle) ? "!" : "1";
            keyDownMsg = (shiftToggle) ? "!" : "1";
            break;

        case '2':
            keyMsg = (shiftToggle) ? "\"" : "2";
            keyDownMsg = (shiftToggle) ? "\"" : "2";
            break;

        case '3':
            keyMsg = (shiftToggle) ? "#" : "3";
            keyDownMsg = (shiftToggle) ? "#" : "3";
            break;

        case '4':
            keyMsg = (shiftToggle) ? "$" : "4";
            keyDownMsg = (shiftToggle) ? "$" : "4";
            break;

        case '5':
            keyMsg = (shiftToggle) ? "%" : "5";
            keyDownMsg = (shiftToggle) ? "%" : "5";
            break;

        case '6':
            keyMsg = (shiftToggle) ? "&" : "6";
            keyDownMsg = (shiftToggle) ? "&" : "6";
            break;

        case '7':
            keyMsg = (shiftToggle) ? "\'" : "7";
            keyDownMsg = (shiftToggle) ? "\'" : "7";
            break;

        case '8':
            keyMsg = (shiftToggle) ? "(" : "8";
            keyDownMsg = (shiftToggle) ? "(" : "8";
            break;

        case '9':
            keyMsg = (shiftToggle) ? ")" : "9";
            keyDownMsg = (shiftToggle) ? ")" : "9";
            break;

        case VK_OEM_1:
            keyMsg = (shiftToggle) ? "*" : ":";
            keyDownMsg = (shiftToggle) ? "*" : ":";
            break;

        case VK_OEM_PLUS:
            keyMsg = (shiftToggle) ? "+" : ";";
            keyDownMsg = (shiftToggle) ? "+" : ";";
            break;

        case VK_OEM_COMMA:
            keyMsg = (shiftToggle) ? "<" : ",";
            keyDownMsg = (shiftToggle) ? "<" : ",";
            break;

        case VK_OEM_MINUS:
            keyMsg = (shiftToggle) ? "=" : "-";
            keyDownMsg = (shiftToggle) ? "=" : "-";
            break;

        case VK_OEM_PERIOD:
            keyMsg = (shiftToggle) ? ">" : ".";
            keyDownMsg = (shiftToggle) ? ">" : ".";
            break;

        case VK_OEM_2:
            keyMsg = (shiftToggle) ? "?" : "/";
            keyDownMsg = (shiftToggle) ? "?" : "/";
            break;

        case VK_OEM_3:
            keyMsg = (shiftToggle) ? "`" : "@";
            keyDownMsg = (shiftToggle) ? "`" : "@";
            break;

        case VK_OEM_4:
            keyMsg = (shiftToggle) ? "{" : "[";
            keyDownMsg = (shiftToggle) ? "{" : "[";
            break;

        case VK_OEM_5:
            keyMsg = (shiftToggle) ? "|" : "\\";
            keyDownMsg = (shiftToggle) ? "|" : "\\";
            break;

        case VK_OEM_6:
            keyMsg = (shiftToggle) ? "}" : "]";
            keyDownMsg = (shiftToggle) ? "}" : "]";
            break;

        case VK_OEM_7:
            keyMsg = (shiftToggle) ? "~" : "^";
            keyDownMsg = (shiftToggle) ? "~" : "^";
            break;

        case VK_OEM_102:
            keyMsg = (shiftToggle) ? "_" : "\\";
            keyDownMsg = (shiftToggle) ? "_" : "\\";
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
            keyMsg = (shiftToggle) ? 
                std::string(1, static_cast<char>(wParam)) : std::string(1, std::tolower(static_cast<char>(wParam)));
            keyDownMsg = (shiftToggle) ? 
                std::string(1, static_cast<char>(wParam)) : std::string(1, std::tolower(static_cast<char>(wParam)));
            break;

    }
}

void Glpa::Scene::getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (wParam)
    {
        case VK_SPACE:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_SPACE;
            break;

        case VK_ESCAPE:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_ESCAPE;
            break;

        case VK_TAB:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_TAB;
            break;

        case VK_BACK:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_BACKSPACE;
            break;

        case VK_LSHIFT:
            shiftToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_LSHIFT;
            break;

        case VK_LCONTROL:
            ctrlToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_LCTRL;
            break;

        case VK_LMENU:
            altToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_LALT;
            break;

        case VK_LWIN:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_LWIN;
            break;

        case VK_RSHIFT:
            shiftToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_RSHIFT;
            break;

        case VK_RCONTROL:
            ctrlToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_RCTRL;
            break;

        case VK_RMENU:
            altToggle = false;
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_RALT;
            break;

        case VK_RWIN:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_RWIN;
            break;

        case VK_F1:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F1;
            break;

        case VK_F2:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F2;
            break;

        case VK_F3:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F3;
            break;

        case VK_F4:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F4;
            break;

        case VK_F5:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F5;
            break;

        case VK_F6:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F6;
            break;

        case VK_F7:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F7;
            break;

        case VK_F8:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F8;
            break;

        case VK_F9:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F9;
            break;

        case VK_F10:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F10;
            break;

        case VK_F11:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F11;
            break;

        case VK_F12:
            keyMsg = "";
            keyUpMsg = Glpa::CHAR_F12;
            break;

        case '0':
            keyMsg = "";
            keyUpMsg = "0";
            break;

        case '1':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "!" : "1";
            break;

        case '2':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "\"" : "2";
            break;

        case '3':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "#" : "3";
            break;

        case '4':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "$" : "4";
            break;

        case '5':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "%" : "5";
            break;

        case '6':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "&" : "6";
            break;

        case '7':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "\'" : "7";
            break;

        case '8':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "(" : "8";
            break;

        case '9':
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? ")" : "9";
            break;

        case VK_OEM_1:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "*" : ":";
            break;

        case VK_OEM_PLUS:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "+" : ";";
            break;

        case VK_OEM_COMMA:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "<" : ",";
            break;

        case VK_OEM_MINUS:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "=" : "-";
            break;

        case VK_OEM_PERIOD:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? ">" : ".";
            break;

        case VK_OEM_2:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "?" : "/";
            break;

        case VK_OEM_3:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "`" : "@";
            break;

        case VK_OEM_4:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "{" : "[";
            break;

        case VK_OEM_5:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "|" : "\\";
            break;

        case VK_OEM_6:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "}" : "]";
            break;

        case VK_OEM_7:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "~" : "^";
            break;

        case VK_OEM_102:
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? "_" : "\\";
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
            keyMsg = "";
            keyUpMsg = (shiftToggle) ? 
                std::string(1, static_cast<char>(wParam)) : std::string(1, std::tolower(static_cast<char>(wParam)));
            break;

    }
}

void Glpa::Scene::getMouse(UINT msg, WPARAM wParam, LPARAM lParam, int dpi)
{
    switch (msg)
    {
        case WM_RBUTTONDOWN:
            mouseRDownPos.x = LOWORD(lParam) * dpi;  
            mouseRDownPos.y = HIWORD(lParam) * dpi;
            mouseRDownPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_RBTN_DOWN;
            break;

        case WM_RBUTTONUP:
            mouseRUpPos.x = LOWORD(lParam) * dpi;  
            mouseRUpPos.y = HIWORD(lParam) * dpi;
            mouseRUpPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_RBTN_UP;
            break;

        case WM_RBUTTONDBLCLK:
            mouseRDbClickPos.x = LOWORD(lParam) * dpi;  
            mouseRDbClickPos.y = HIWORD(lParam) * dpi;
            mouseRDbClickPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_RBTN_DBCLICK;
            break;

        case WM_LBUTTONDOWN:
            mouseLDownPos.x = LOWORD(lParam) * dpi;  
            mouseLDownPos.y = HIWORD(lParam) * dpi;
            mouseLDownPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_LBTN_DOWN;
            break;

        case WM_LBUTTONUP:
            mouseLUpPos.x = LOWORD(lParam) * dpi;  
            mouseLUpPos.y = HIWORD(lParam) * dpi;
            mouseLUpPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_LBTN_UP;
            break;

        case WM_LBUTTONDBLCLK:
            mouseLDbClickPos.x = LOWORD(lParam) * dpi;  
            mouseLDbClickPos.y = HIWORD(lParam) * dpi;
            mouseLDbClickPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_LBTN_DBCLICK;
            break;
            
        case WM_MBUTTONDOWN:
            mouseMDownPos.x = LOWORD(lParam) * dpi;  
            mouseMDownPos.y = HIWORD(lParam) * dpi;
            mouseMDownPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_MBTN_DOWN;
            break;
            
        case WM_MBUTTONUP:
            mouseMUpPos.x = LOWORD(lParam) * dpi;  
            mouseMUpPos.y = HIWORD(lParam) * dpi;
            mouseMUpPos.fill();
            mouseMsg = Glpa::CHAR_MOUSE_MBTN_UP;
            break;
            
        case WM_MOUSEWHEEL:
            wheelMoveAmount = GET_WHEEL_DELTA_WPARAM(wParam);
            mouseMsg = Glpa::CHAR_MOUSE_WHEEL;
            break;

        case WM_MOUSEMOVE:
            mousePos.x = LOWORD(lParam) * dpi;  
            mousePos.y = HIWORD(lParam) * dpi;
            mouseMsg = Glpa::CHAR_MOUSE_MOVE;
            break;
    }
}

std::string Glpa::Scene::GetNowKeyMsg()
{
    return keyMsg;
}

bool Glpa::Scene::GetNowKeyMsg(std::string argMsg)
{
    if (keyMsg == argMsg) return true;
    else return false;
}

std::string Glpa::Scene::GetNowKeyDownMsg()
{
    return keyDownMsg;
}

bool Glpa::Scene::GetNowKeyDownMsg(std::string argMsg)
{
    if (keyDownMsg == argMsg) return true;
    else return false;
}

std::string Glpa::Scene::GetNowKeyUpMsg()
{
    return keyUpMsg;
}

bool Glpa::Scene::GetNowKeyUpMsg(std::string argMsg)
{
    if (keyUpMsg == argMsg) return true;
    else return false;
}

void Glpa::Scene::updateKeyMsg()
{
    keyDownMsg = "";
    keyUpMsg = "";
}

void Glpa::Scene::updateMouseMsg()
{
    mouseMsg = "";
    mouseRDownPos.empty();
    mouseRDbClickPos.empty();
    mouseRUpPos.empty();

    mouseLDownPos.empty();
    mouseLDbClickPos.empty();
    mouseLUpPos.empty();

    mouseMDownPos.empty();
    mouseMUpPos.empty();

    wheelMoveAmount = 0;
}

std::string Glpa::Scene::GetNowMouseMsg()
{
    return mouseMsg;
}

bool Glpa::Scene::GetNowMouseMsg(std::string argMsg)
{
    if (mouseMsg == argMsg) return true;
    else return false;
}

bool Glpa::Scene::GetNowMouseMsg(std::string argMsg, Glpa::Vec2d &target)
{
    if (argMsg != keyMsg) return false;

    if (argMsg == Glpa::CHAR_MOUSE_MOVE)
    {
        target = mousePos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_RBTN_DOWN)
    {
        target = mouseRDownPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_RBTN_UP)
    {
        target = mouseRUpPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_RBTN_DBCLICK)
    {
        target = mouseRDbClickPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_LBTN_DOWN)
    {
        target = mouseLDownPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_LBTN_UP)
    {
        target = mouseLUpPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_LBTN_DBCLICK)
    {
        target = mouseLDbClickPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_MBTN_DOWN)
    {
        target = mouseMDownPos;
        return true;
    }
    else if (argMsg == Glpa::CHAR_MOUSE_MBTN_UP)
    {
        target = mouseMUpPos;
        return true;
    }
    else
    {
        return false;
    }
}

bool Glpa::Scene::GetNowMouseMsg(std::string argMsg, int &amount)
{
    if (argMsg != keyMsg) return false;

    if (argMsg == Glpa::CHAR_MOUSE_WHEEL)
    {
        amount = wheelMoveAmount;
        return true;
    }
    else
    {
        return false;
    }
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
        if (obj.second->isLoaded()) obj.second->release();
    }
}
