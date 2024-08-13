#include "GlpaEvent.h"
#include "GlpaLog.h"

Glpa::EventManager* Glpa::EventManager::instance = nullptr;

void Glpa::EventManager::Create()
{
    if (instance == nullptr)
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "");
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

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + eventList->getTag() + "]");
    instance->eventLists[eventList->getTag()] = eventList;
}

bool Glpa::EventManager::ExecuteEvent(std::string eventListStr)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, eventListStr);
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
            Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + eventTag + "]");
            return instance->eventLists[eventTag]->execute("", "");
        }
        
        // If there is a function which has no arguments in the event string, execute the event without arguments.
        if (eventStr.find_first_of(' ') == std::string::npos)
        {
            std::string eventName = eventStr;
            std::string argStr = "";
            Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + eventTag + "] Name[" + eventName + "]");
            return instance->eventLists[eventTag]->execute(eventName, argStr);
        }

        // If there is a function which has arguments in the event string, execute the event with arguments.
        std::string eventName = eventStr.substr(0, eventStr.find_first_of(' '));
        std::string argStr = eventStr.substr(eventStr.find_first_of(' ') + 1, eventStr.size() - eventStr.find_first_of(' '));
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + eventTag + "] Name[" + eventName + "] Arg[" + argStr + "]");
        return instance->eventLists[eventTag]->execute(eventName, argStr);
    }

    return false;
}

void Glpa::EventManager::Release()
{
    if (instance != nullptr)
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "");
        for (auto it = instance->eventLists.begin(); it != instance->eventLists.end(); it++)
        {
            (*it).second->release();

            Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete event list[" + it->first + "]");
            delete it->second;
            it->second = nullptr;
        }
        
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete manager instance");
        delete instance;
        instance = nullptr;
    }
}

Glpa::Event::Event
(
    std::string argName, const char *fileChar, int lineNum,
    std::initializer_list<std::vector<std::string>> argCdsList
){
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor[" + argName + "]");
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Event Name[" + name + "] arg filter : " + name);
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
                Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Event Name[" + name + "] arg filter failed : " + name);
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
                Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Event Name[" + name + "] arg filter failed : " + name);
                return false;
            }
        }
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Event Name[" + name + "] arg filter success : " + name);
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
            "EventList : " + tag,
            "Event name : " + event->getName()
            }
        );
    }

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + tag + "] add : " + event->getName());
    events[event->getName()] = event;
}

bool Glpa::EventList::execute(std::string eventName, std::string args)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + tag + "] execute : " + eventName);
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
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "EventList[" + tag + "]");
    for (auto it = events.begin(); it != events.end(); it++)
    {
        Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Delete event[" + it->first + "]");
        delete it->second;
        it->second = nullptr;
    }
}
