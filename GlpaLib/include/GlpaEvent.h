#ifndef GLPA_EVENT_H_
#define GLPA_EVENT_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <initializer_list>

#include "ErrorHandler.h"

namespace Glpa
{

class Event
{
private :
    std::string name = "";
    std::vector<std::string> argTypes;
    std::vector<std::string> args;

public :
    Event
    (
        std::string argName,const char* fileChar, int lineNum, 
        std::initializer_list<std::string> typeList,
        std::initializer_list<std::string> argList
    );
    virtual ~Event(){};

    std::string file;
    int line;

    std::string getName() const {return name;}
    virtual void onEvent(std::vector<std::string> args) = 0;
};

class EventList
{
private :
    std::string tag = "";
    std::unordered_map<std::string, Glpa::Event*> events;

public :
    EventList(std::string argTag) : tag(argTag){};
    ~EventList(){};

    void AddEvent(Glpa::Event* event);

    std::string getTag() const {return tag;}
    void execute(std::string eventName, std::string args);

    void release();
};

class EventManager
{
private :
    static Glpa::EventManager* instance;
    std::unordered_map<std::string, Glpa::EventList*> eventLists;
    EventManager(){};

public :
    ~EventManager(){};

    static void Create();
    static void AddEvent(Glpa::EventList* eventList);
    static bool ExecuteEvent(std::string eventListStr);

    static void Release();
};

}

#endif GLPA_EVENT_H_