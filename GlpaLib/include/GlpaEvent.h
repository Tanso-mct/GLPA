#ifndef GLPA_EVENT_H_
#define GLPA_EVENT_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <initializer_list>

#include "ErrorHandler.cuh"

namespace Glpa
{

class Event
{
private :
    std::string name = "";
    std::vector<std::vector<std::string>> argCds; // Argument Candidates

public :
    Event
    (
        std::string argName, const char* fileChar, int lineNum,
        std::initializer_list<std::vector<std::string>> argCdsList
    );

    virtual ~Event()
    {
        Glpa::OutputLog
        (
            __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor[" + name + "]"
        );
    };

    std::string file;
    int line;

    std::string getName() const {return name;}
    virtual bool onEvent(std::vector<std::string> args) = 0;

    bool argFilter(std::vector<std::string> args);
};

class EventList
{
private :
    std::string tag = "";
    std::unordered_map<std::string, Glpa::Event*> events;

public :
    EventList(std::string argTag) : tag(argTag)
    {
        Glpa::OutputLog
        (
            __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor[" + tag + "]"
        );
    };

    virtual ~EventList()
    {
        Glpa::OutputLog
        (
            __FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor[" + tag + "]"
        );
    };

    void AddEvent(Glpa::Event* event);

    std::string getTag() const {return tag;}
    bool execute(std::string eventName, std::string args);

    void release();
};

class EventManager
{
private :
    static Glpa::EventManager* instance;
    std::unordered_map<std::string, Glpa::EventList*> eventLists;
    EventManager()
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    };

public :
    ~EventManager()
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
    };

    static void Create();
    static void AddEventList(Glpa::EventList* eventList);
    static bool ExecuteEvent(std::string eventListStr);

    static void Release();
};

}

#endif GLPA_EVENT_H_