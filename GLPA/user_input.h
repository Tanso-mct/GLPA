#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "scene.h"
#include "text.h"

#include "error.h"

#define GLPA_USERINPUT_MESSAGE_KEYDOWN 0
#define GLPA_USERINPUT_MESSAGE_KEYUP 1
#define GLPA_USERINPUT_MESSAGE_MOUSEMOVE 2
#define GLPA_USERINPUT_MESSAGE_MOUSELBTNDOWN 3
#define GLPA_USERINPUT_MESSAGE_MOUSELBTNUP 4
#define GLPA_USERINPUT_MESSAGE_MOUSELBTNDBCLICK 5
#define GLPA_USERINPUT_MESSAGE_MOUSERBTNDOWN 6
#define GLPA_USERINPUT_MESSAGE_MOUSERBTNUP 7
#define GLPA_USERINPUT_MESSAGE_MOUSERBTNDBCLICK 8
#define GLPA_USERINPUT_MESSAGE_MOUSEMBTNDOWN 9
#define GLPA_USERINPUT_MESSAGE_MOUSEMBTNUP 10
#define GLPA_USERINPUT_MESSAGE_MOUSEMBTNDBWHEEL 11

#define GLPA_USER_FUNC std::function<void(std::string, UINT, WPARAM, LPARAM)>

class UserInput
{
public :
    void add(
        std::wstring func_name, 
        GLPA_USER_FUNC add_func , 
        HWND get_message_window_handle, 
        int message_type
    );
    
    void edit(std::wstring func_name, GLPA_USER_FUNC edited_func);

    void eraseFunc(std::wstring func_name, std::unordered_map<HWND, std::vector<std::wstring>>* arg_msg_func);
    void release(std::wstring func_name);

    // Input
    std::wstring convertWParamToWstr();

    // Key message
    void keyDown(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void keyUp(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);

    void keyTypingScene2d(Scene2d* pt_scene_2d, WPARAM w_param);
    
    // Mouse Move message
    void mouseMove(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    
    // Mouse Left button message
    void mouseLbtnDown(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseLbtnUp(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseLbtnDblclick(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);

    // Mouse Right button message
    void mouseRbtnDown(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseRbtnUp(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseRbtnDblClick(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);

    // Mouse Middle button message
    void mouseMbtnDown(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseMbtnUp(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void mouseMbtnWheel(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);

    bool typing = false;

private :
    std::unordered_map<std::wstring, GLPA_USER_FUNC> myFunc;
    std::unordered_map<std::wstring, std::vector<int>> msgFunc;

    std::unordered_map<HWND, std::vector<std::wstring>> keyDownFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> keyUpFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseMoveFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseLbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseLbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseLbtnDblclickFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseRbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseRbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseRbtnDblClickFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseMbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseMbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::wstring>> mouseMbtnWheelFunc;

};

#endif USERINPUT_H_

