#ifndef GLPA_BASE_H_
#define GLPA_BASE_H_

#include <string>
#include <unordered_map>

#include "Window.h"
#include "Scene.h"

class GlpaBase
{
private :
    std::string name;
    bool visible = true;
    bool started = false;

    Glpa::Window* window;

    std::string nowScName;
    std::unordered_map<std::string, Glpa::Scene*> pScs;
    
public :
    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    bool getVisible() const {return visible;}
    void setVisible(bool value) {visible = value;}

    bool getStarted() const {return started;}
    void setStarted(bool value) {started = value;}

    Glpa::Scene* getNowScenePt(){return pScs[nowScName];}

    virtual void setup() = 0;



    void start();
    void update();

};

#endif GLPA_BASE_H_
