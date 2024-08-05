#ifndef GLPA_EVENT_H_
#define GLPA_EVENT_H_

#include <string>
#include <unordered_map>
#include "ErrorHandler.h"

namespace Glpa
{

class Event
{
private :
    std::string name = "";
public :
    Event(std::string argName) : name(argName){};
    virtual ~Event(){};

    std::string getName() const {return name;}
    virtual void onEvent() = 0;
};

class EventManager
{
private :
    static Glpa::EventManager* instance;
    std::unordered_map<std::string, Glpa::Event*> events;
    EventManager(){};

public :
    ~EventManager(){};

    static void Create();
    static void AddEvent(Glpa::Event* event);
    static bool ExecuteEvent(std::string eventName);

    static void Release();
};

}

#endif GLPA_EVENT_H_