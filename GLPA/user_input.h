#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <Windows.h>
#include <string>
#include <vector>
#include <unordered_map>

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

class UserInput
{
public :
    void add(
        std::string func_name, 
        void(*pt_add_func)(
            std::string scene_name, 
            WPARAM w_param, 
            LPARAM l_param
        ) , 
        HWND get_message_window_handle, 
        int message_type
    );
    
    void edit();
    void remove(std::wstring func_name);

    //Key message
    void keyDown(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void keyUp(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    
    //Mouse Move message
    void mouseMove(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    
    //Mouse Left button message
    void mouseLbtnDown(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseLbtnUp(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseLbtnDblclick(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);

    //Mouse Right button message
    void mouseRbtnDown(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseRbtnUp(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseRbtnDblClick(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);

    //Mouse Middle button message
    void mouseMbtnDown(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseMbtnUp(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);
    void mouseMbtnWheel(HWND h_wnd, std::string scene_name, WPARAM w_param, LPARAM l_param);

private :
    std::unordered_map<std::string, void(*)(std::string scene_name, WPARAM w_param, LPARAM l_param)> myFunc;
    bool typing = false;

    std::unordered_map<HWND, std::vector<std::string>> keyDownFunc;
    std::unordered_map<HWND, std::vector<std::string>> keyUpFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseMoveFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseLbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseLbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseLbtnDblclickFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseRbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseRbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseRbtnDblClickFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseMbtnDownFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseMbtnUpFunc;
    std::unordered_map<HWND, std::vector<std::string>> mouseMbtnWheelFunc;

};

#endif USERINPUT_H_

