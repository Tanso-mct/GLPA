#ifndef COMMAND_H_
#define COMMAND_H_

#include <functional>
#include <unordered_map>
#include <string>

#define COMMAND_FUN_FUNCTIONAL std::function<void()>

#define COMMAND_TAG_C L"/c "
#define COMMAND_TAG_C_SIZE 3


#define COMMAND_CREATE_GAME_WINDOW L"create_game_window"
#define COMMAND_LOAD_GAME_SCENE "load_game_scene"

class Command{
public :
    void execute(std::wstring command_name);

private :
    std::unordered_map<std::wstring, COMMAND_FUN_FUNCTIONAL> list;
};

#endif COMMAND_H_


