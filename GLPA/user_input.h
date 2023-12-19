#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <Windows.h>
#include <string>
#include <unordered_map>

class UserInput
{
public :
    void add();
    void edit();
    void remove();

    void input(HWND h_wnd, UINT msg, WPARAM w_param, LPARAM l_param);

private :
    std::unordered_map<int, std::string> msgDoMyFunc;
    std::unordered_map<std::string, void(*)()> myFunc;
};

#endif USERINPUT_H_

