#ifndef GLPA_SCENE_OBJECT_H_
#define GLPA_SCENE_OBJECT_H_

#include <string>
#include <locale>
#include <codecvt>

#include "Vector.h"
#include "FileDataManager.h"

namespace Glpa
{

typedef struct _OBJECT3D_DATA
{
    int id;
    int mtId;
    Glpa::RANGE_RECT range;
    Glpa::POLYGON* polygons;
} OBJECT3D_DATA;

typedef struct _OBJECT_INFO
{
    bool isVisible;
    float pos[3];
    float rot[3];
    float scale[3];
} OBJECT_INFO;

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
