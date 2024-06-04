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

    // You must enter the name of the first scene in startScName in the Awake function.
    std::string startScName;
    std::string nowScName;
    std::string nextScName;

    std::unordered_map<std::string, Glpa::Scene*> ptScs;
    
public :
    Glpa::Window* window;

    GlpaBase();
    ~GlpaBase();
    
    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    bool getVisible() const {return visible;}
    void setVisible(bool value) {visible = value;}

    bool getStarted() const {return started;}
    void setStarted(bool value) {started = value;}

    Glpa::Scene* getNowScenePt(){return ptScs[nowScName];}

    void AddScene(Glpa::Scene* ptScene);
    void DeleteScene(Glpa::Scene* ptScene);

    void loadScene();
    void loadScene(Glpa::Scene* ptScene);

    void releaseScene();
    void releaseScene(Glpa::Scene* ptScene);

    void releaseAllScene();

    void setFirstSc(Glpa::Scene* ptScene);

    /// @brief Create scene data and set the name of the first scene.
    virtual void setup() = 0;

    virtual void awake(){};
    virtual void destroy(){};

    virtual void start(){};
    virtual void update(){};

    void runStart();
    void runUpdate();

};

#endif GLPA_BASE_H_
