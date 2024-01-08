#ifndef COMMAND_H_
#define COMMAND_H_

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

#include "error.h"

#define COMMAND_FUN_FUNCTIONAL std::function<void()>

#define COMMAND_FUN_PT(instance, method_name) \
    [&instance](){ \
        instance.method_name(); \
    }

#define COMMAND_TAG_C L"/c "
#define COMMAND_TAG_C_SIZE 3

class Command{
public :
    void add(std::wstring func_name, COMMAND_FUN_FUNCTIONAL pt_add_func);
    void edit(std::wstring func_name, COMMAND_FUN_FUNCTIONAL pt_edit_func);
    void release(std::wstring func_name);

    void execute(std::wstring command_name);

private :
    std::unordered_map<std::wstring, COMMAND_FUN_FUNCTIONAL> list;
};

#endif COMMAND_H_


