#ifndef USERINPUT_H_
#define USERINPUT_H_

#include <string>
#include <unordered_map>

class UserInput
{
public :
    void add();
    void edit();
    void remove();

private :
    std::unordered_map<int, std::string> msgDoMyFunc;
    std::unordered_map<std::string, void(*)()> myFunc;
};

#endif USERINPUT_H_

