#include "GlpaEvent.h"
#include "GlpaLog.h"

Glpa::EventManager* Glpa::EventManager::instance = nullptr;

void Glpa::EventManager::Create()
{
    if (instance == nullptr)
    {
        Glpa::OutputLog(__FILE__, __LINE__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventManager Manager created");
        instance = new Glpa::EventManager();
    }
}

void Glpa::EventManager::AddEventList(Glpa::EventList *eventList)
{
    if (instance->eventLists.find(eventList->getTag()) != instance->eventLists.end())
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__, 
            {"Event list already exists.",
            "Event list tag : " + eventList->getTag()
            }
        );
    }

    instance->eventLists[eventList->getTag()] = eventList;
}

bool Glpa::EventManager::ExecuteEvent(std::string eventListStr)
{
    bool hasFunc = false;
    if (eventListStr.find_first_of(' ') != std::string::npos) hasFunc = true;

    std::string eventTag;
    std::string eventStr;
    if (!hasFunc)
    {
        eventTag = eventListStr;
    }
    else
    {
        eventTag = eventListStr.substr(0, eventListStr.find_first_of(' '));
        eventStr  = eventListStr.substr
        (
            eventListStr.find_first_of(' ') + 1, eventListStr.size() - eventListStr.find_first_of(' ')
        );
    }

    if (instance->eventLists.find(eventTag) != instance->eventLists.end())
    {
        // If there is no function in the event string, execute the event without function.
        if (!hasFunc)
        {
            return instance->eventLists[eventTag]->execute("", "");
        }
        
        // If there is a function which has no arguments in the event string, execute the event without arguments.
        if (eventStr.find_first_of(' ') == std::string::npos)
        {
            std::string eventName = eventStr;
            std::string argStr = "";
            return instance->eventLists[eventTag]->execute(eventName, argStr);
        }

        // If there is a function which has arguments in the event string, execute the event with arguments.
        std::string eventName = eventStr.substr(0, eventStr.find_first_of(' '));
        std::string argStr = eventStr.substr(eventStr.find_first_of(' ') + 1, eventStr.size() - eventStr.find_first_of(' '));
        return instance->eventLists[eventTag]->execute(eventName, argStr);
    }

    return false;
}

void Glpa::EventManager::Release()
{
    if (instance != nullptr)
    {
        for (auto it = instance->eventLists.begin(); it != instance->eventLists.end(); it++)
        {
            (*it).second->release();
            delete it->second;
            it->second = nullptr;
        }
        
        delete instance;
        instance = nullptr;
    }
}

Glpa::Event::Event
(
    std::string argName, const char *fileChar, int lineNum,
    std::initializer_list<std::vector<std::string>> argCdsList
){
    name = argName;
    file = fileChar;
    line = lineNum;

    for (std::vector<std::string> args : argCdsList)
    {
        std::vector<std::string> argVec;
        for (int i = 0; i < args.size(); i++)
        {
            argVec.push_back(args[i]);
        }
        argCds.push_back(argVec);
    }
}

bool Glpa::Event::argFilter(std::vector<std::string> args)
{
    bool isSameSize = (args.size() == argCds.size());
    bool isOneMinSize = false;
    if (argCds[argCds.size() - 1].size() == 0 && args.size() == argCds.size() - 1)
    {
        isOneMinSize = true;
    }

    if (!isSameSize && !isOneMinSize) return false;

    if (isSameSize)
    {
        for (int i = 0; i < args.size(); i++)
        {
            if (argCds[i].size() == 0) continue;
            if (std::find(argCds[i].begin(), argCds[i].end(), args[i]) == argCds[i].end())
            {
                return false;
            }
        }
    }

    if (isOneMinSize)
    {
        for (int i = 0; i < args.size() - 1; i++)
        {
            if (argCds[i].size() == 0) continue;
            if (std::find(argCds[i].begin(), argCds[i].end(), args[i]) == argCds[i].end())
            {
                return false;
            }
        }
    }

    return true;
}

void Glpa::EventList::AddEvent(Glpa::Event *event)
{
    if (events.find(event->getName()) != events.end())
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__, 
            {"Event already exists.",
            "Tag : " + tag,
            "Event name : " + event->getName()
            }
        );
    }

    events[event->getName()] = event;
}

bool Glpa::EventList::execute(std::string eventName, std::string args)
{
    if (events.find(eventName) == events.end())
    {
        return false;
    }

    std::vector<std::string> argList;
    std::string arg = "";
    for (int i = 0; i < args.size(); i++)
    {
        if (args[i] == ' ')
        {
            argList.push_back(arg);
            arg = "";
        }
        else
        {
            arg += args[i];
        }
    }

    argList.push_back(arg);

    if (!events[eventName]->argFilter(argList))
    {
        return false;
    }

    return events[eventName]->onEvent(argList);
}

void Glpa::EventList::release()
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        delete it->second;
        it->second = nullptr;
    }
}
