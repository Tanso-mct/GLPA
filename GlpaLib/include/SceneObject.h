#ifndef GLPA_SCENE_OBJECT_H_
#define GLPA_SCENE_OBJECT_H_

#include <string>
#include <locale>
#include <codecvt>

#include "Vector.cuh"
#include "FileDataManager.h"

namespace Glpa
{

class SceneObject
{
protected :
    std::string type;
    std::string name;
    std::string filePath;
    bool loaded = false;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> strConverter;

    Glpa::FileDataManager* fileDataManager = nullptr;

public :
    SceneObject();
    virtual ~SceneObject();

    void setManager(Glpa::FileDataManager* argFileDataManager);

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    std::string getType() const {return type;}
    void setType(std::string str) {type = str;}

    bool isLoaded() const {return loaded;}

    virtual void load() = 0;
    virtual void release() = 0;
};

}

#endif GLPA_SCENE_OBJECT_H_
