#ifndef GLPA_EVENT_H_
#define GLPA_EVENT_H_

#include <string>

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

}

#endif GLPA_EVENT_H_