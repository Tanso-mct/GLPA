#include "GlpaEvent.h"

Glpa::EventManager* Glpa::EventManager::instance = nullptr;

void Glpa::EventManager::Create()
{
    if (instance == nullptr)
    {
        instance = new Glpa::EventManager();
    }
}

void Glpa::EventManager::AddEvent(Glpa::Event *event)
{
    if (instance->events.find(event->getName()) != instance->events.end())
    {
        Glpa::runTimeError(__FILE__, __LINE__, "Event already exists.");
    }

    instance->events[event->getName()] = event;
}

bool Glpa::EventManager::ExecuteEvent(std::string eventName)
{
    if (instance->events.find(eventName) != instance->events.end())
    {
        instance->events[eventName]->onEvent();
        return true;
    }

    return false;
}

void Glpa::EventManager::Release()
{
    if (instance != nullptr)
    {
        for (auto it = instance->events.begin(); it != instance->events.end(); it++)
        {
            delete it->second;
            it->second = nullptr;
        }
        
        delete instance;
        instance = nullptr;
    }
}
