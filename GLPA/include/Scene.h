#ifndef GLPA_SCENE_H_
#define GLPA_SCENE_H_

#include <string>


namespace Glpa{

class Scene
{
private :
    std::string name;

protected :
    std::string keyMsg;
    bool keyMsgUpdated = false;

    std::string mouseMsg;
    bool mouseMsgUpdated = false;

public :
    Scene();
    ~Scene();
    
    void getKeyDown(UINT msg, WPARAM wParam, LPARAM lParam);
    void getKeyUp(UINT msg, WPARAM wParam, LPARAM lParam);
    void getMouse(UINT msg, WPARAM wParam, LPARAM lParam);

    virtual void start() = 0;
    virtual void update() = 0;

    virtual void awake() = 0;
    virtual void destroy() = 0;

};

}

#endif GLPA_SCENE_H_
