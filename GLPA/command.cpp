#include "command.h"


void Command::add(std::wstring funcName, COMMAND_FUN_FUNCTIONAL ptAddFunc){
    list[funcName] = ptAddFunc;
}


void Command::edit(std::wstring funcName, COMMAND_FUN_FUNCTIONAL ptEditFunc){
    if (list.find(funcName) != list.end()){
        list[funcName] = ptEditFunc;
    }
    else{
        std::runtime_error(ERROR_COMMAND_LIST);
    }
}


void Command::release(std::wstring funcName){
    if (list.find(funcName) != list.end()){
        list.erase(funcName);
    }
    else{
        std::runtime_error(ERROR_COMMAND_LIST);
    }
}


void Command::execute(std::wstring commandName){
    if (list.find(commandName) != list.end()){
        list[commandName]();
    }
    else{
        std::runtime_error(ERROR_COMMAND_LIST);
    }
}
