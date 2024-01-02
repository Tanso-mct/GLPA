#ifndef COMMAND_H_
#define COMMAND_H_

#include <functional>
#include <unordered_map>
#include <string>

#define COMMAND_FUN_FUNCTIONAL std::function<void()>

class Command{
public :
    void execute(std::wstring command_name);

private :
    std::unordered_map<std::wstring, COMMAND_FUN_FUNCTIONAL> list;
};

#endif COMMAND_H_


