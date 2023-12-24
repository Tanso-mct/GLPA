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

#define GLPA_USERINPUT_ADD_CASE(msg_name, msg_func_data) \
    case msg_name: \
        myFunc[funcName] = ptAddFunc; \
        msgFunc[funcName].push_back(msg_name); \
        msg_func_data[getMsgWndHwnd].push_back(funcName); \
        break;

#define GLPA_USERINPUT_RELEASE_CASE(msg_name, msg_func_data) \
    case msg_name: \
        eraseFunc(funcName, &msg_func_data); \
        break; 


#define GLPA_USER_FUNC_FUNCTIONAL std::function<void(std::string, UINT, WPARAM, LPARAM)>

#define GLPA_USER_FUNC_PT(instance, method_name) \
    [&instance](std::string scene_name, UINT msg, WPARAM wParam, LPARAM lParam) { \
        instance.method_name(scene_name, msg, wParam, lParam); \
    }

#define GLPA_USER_FUNC(method_name) \
    void method_name(std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param)

#define GLPA_USER_FUNC_DEFINE(class_name, method_name, scene_name_arg_name, msg_arg_name, w_param_arg_name, l_param_arg_name) \
    void class_name::method_name(std::string scene_name_arg_name, UINT msg_arg_name, WPARAM w_param_arg_name, LPARAM l_param_arg_name)


#define GLPA_USERINPUT_MSG_FUNC_DEFINE(method_name, msg_func_data) \
    void UserInput::method_name(HWND hWnd, std::string scName, UINT msg, WPARAM wParam, LPARAM lParam){ \
        for (auto funcName : msg_func_data[hWnd]){ \
            myFunc[funcName](scName, msg, wParam, lParam); \
        } \
    }

class UserInput
{
public :
    void add(
        std::wstring func_name, 
        GLPA_USER_FUNC_FUNCTIONAL add_func , 
        HWND get_message_window_handle, 
        int message_type
    );
    
    void edit(std::wstring func_name, GLPA_USER_FUNC_FUNCTIONAL edited_func);

    void eraseFunc(std::wstring func_name, std::unordered_map<HWND, std::vector<std::wstring>>* arg_msg_func);
    void release(std::wstring func_name);

    // convert WPARAM to std::wstring
    std::wstring convertWParamToLowWstr(WPARAM w_param);

    // Key message
    void keyDown(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);
    void keyUp(HWND h_wnd, std::string scene_name, UINT msg, WPARAM w_param, LPARAM l_param);

    void keyDownTypingScene2d(
        Scene2d* pt_scene_2d, std::wstring text_group_name, WPARAM w_param, std::wstring w_param_wstr
    );
    void keyUpTypingScene2d(
        Scene2d* pt_scene_2d, std::wstring text_group_name, WPARAM w_param, std::wstring w_param_wstr
    );
    
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
    bool shift = false;

private :
    std::unordered_map<std::wstring, GLPA_USER_FUNC_FUNCTIONAL> myFunc;
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

