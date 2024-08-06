#include "GlpaEvent.h"

Glpa::EventManager* Glpa::EventManager::instance = nullptr;

void Glpa::EventManager::Create()
{
    if (instance == nullptr)
    {
        instance = new Glpa::EventManager();
    }
}

void Glpa::EventManager::AddEvent(Glpa::EventList *eventList)
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
    std::string eventTag = eventListStr.substr(0, eventListStr.find_first_of(' '));
    std::string eventStr = eventListStr.substr(eventListStr.find_first_of(' ') + 1, eventListStr.size() - eventListStr.find_first_of(' '));
    if (instance->eventLists.find(eventTag) != instance->eventLists.end())
    {
        std::string eventName = eventStr.substr(0, eventStr.find_first_of(' '));
        std::string argStr = eventStr.substr(eventStr.find_first_of(' ') + 1, eventStr.size() - eventStr.find_first_of(' '));
        instance->eventLists[eventTag]->execute(eventName, argStr);
        return true;
    }

    return false;
}

void Glpa::EventManager::Release()
{
    if (instance != nullptr)
    {
        for (auto it = instance->eventLists.begin(); it != instance->eventLists.end(); it++)
        {
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
    std::initializer_list<std::string> typeList,
    std::initializer_list<std::string> argList
){
    name = argName;
    file = fileChar;
    line = lineNum;

    for (auto it = typeList.begin(); it != typeList.end(); it++)
    {
        argTypes.push_back(*it);
    }

    for (auto it = argList.begin(); it != argList.end(); it++)
    {
        args.push_back(*it);
    }
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

void Glpa::EventList::execute(std::string eventName, std::string args)
{
    if (events.find(eventName) == events.end())
    {
        Glpa::runTimeError
        (
            __FILE__, __LINE__, 
            {"Event does not exist.",
            "Tag : " + tag,
            "Event name : " + eventName
            }
        );
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

    events[eventName]->onEvent(argList);
}

void Glpa::EventList::release()
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        delete it->second;
        it->second = nullptr;
    }
}
