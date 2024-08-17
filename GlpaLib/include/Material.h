#ifndef GLPA_MATERIAL_H_
#define GLPA_MATERIAL_H_

#include <string>

#include "Constant.h"
#include "FileDataManager.h"

namespace Glpa
{

class Material
{
private :
    bool loaded = false;
    std::string name;

    // Diffuse texture
    std::string diffuseFilePath;

    // Occlusion Roughness Metallic texture
    std::string ormFilePath;

    // Normal texture
    std::string normalFilePath;

    Glpa::FileDataManager* fileDataManager = nullptr;

public :
    Material(std::string argName, std::string argDiffuseFilePath);
    ~Material();

    bool isLoaded() const {return loaded;}

    void setManager(Glpa::FileDataManager* argFileDataManager);

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    int GetMtWidth(std::string mtName);
    int GetMtHeight(std::string mtName);
    LPDWORD GetMtData(std::string mtName);

    void load();
    void release();



};


}



#endif GLPA_MATERIAL_H_