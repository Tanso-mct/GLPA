#ifndef GLPA_BASE_H_
#define GLPA_BASE_H_

#include <string>
#include <unordered_map>

#include "Window.h"
#include "Scene.h"
#include "FileDataManager.h"

class GlpaBase
{
private :
    std::string name;

    /// @brief Indicates whether the window is visible.
    bool visible = true;

    /// @brief Indicates whether the drawing loop has started. true if it has started;
    bool started = false;

    /// @brief You must enter the name of the first scene in startScName in the Awake function.
    std::string startScName;

    /// @brief Name of the currently loaded scene.
    std::string nowScName;

    /// @brief Save scene data.
    std::unordered_map<std::string, Glpa::Scene*> ptScs;

    Glpa::FileDataManager* fileDataManager;
    
public :
    Glpa::Window* window;

    GlpaBase();
    virtual ~GlpaBase();
    
    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    bool getVisible() const {return visible;}
    void setVisible(bool value) {visible = value;}

    bool getStarted() const {return started;}
    void setStarted(bool value) {started = value;}

    void setManager(Glpa::FileDataManager* argFileDataManager){fileDataManager = argFileDataManager;}

    /// @brief Get the pointer to the currently loaded scene.
    /// @return Pointer type for the currently loaded scene.
    Glpa::Scene* getNowScenePt(){return ptScs[nowScName];}

    /// @brief Add a scene. Use within the setup function.
    /// @param ptScene A pointer to a class whose base class is the Scene class.
    void AddScene(Glpa::Scene* ptScene);

    /// @brief Delete the current scene.
    void DeleteScene();
    
    /// @brief Delete added scenes.
    /// @param ptScene A pointer to a class whose base class is the Scene class.
    void DeleteScene(Glpa::Scene* ptScene);

    /// @brief Delete all currently loaded scenes.
    void DeleteAllScene();


    /// @brief Load the first scene.
    void LoadScene();

    /// @brief Load the specified scene.
    /// @param ptScene A pointer to a class whose base class is the Scene class.
    void LoadScene(Glpa::Scene* ptScene);

    /// @brief Release the current scene.
    void ReleaseScene();

    /// @brief Release the specified scene.
    /// @param ptScene A pointer to a class whose base class is the Scene class.
    void ReleaseScene(Glpa::Scene* ptScene);

    /// @brief Release all currently loaded scenes.
    void ReleaseAllScene();


    /// @brief Setting the first scene. This must be done in the setup function.
    /// @param ptScene A pointer to a class whose base class is the Scene class.
    void SetFirstSc(Glpa::Scene* ptScene);


    /// @brief Create scene data and set the name of the first scene.
    virtual void setup() = 0;

    /// @brief Describe the processing that needs to be done in the constructor of Glpa base class.
    virtual void awake(){};

    /// @brief Describe the processing that needs to be done in the destructor of the Glpa base class.
    virtual void destroy(){};

    /// @brief Describe the first process that needs to be done when the drawing loop starts.
    virtual void start(){};

    /// @brief Describe the processing that needs to be done every frame after the loop starts.
    virtual void update(){};
    

    /// @brief Executed at the beginning of the drawing loop.
    void runStart();

    /// @brief Executed every frame in a drawing loop.
    void runUpdate();

};

#endif GLPA_BASE_H_
