#ifndef GLPA_MATERIAL_H_
#define GLPA_MATERIAL_H_

#include <string>

#include "Constant.h"
#include "FileDataManager.h"

namespace Glpa
{

typedef struct _GPU_MATERIAL
{
    LPDWORD baseColor;
} GPU_MATERIAL;

class Material
{
private :
    bool loaded = false;
    std::string name;

    // Diffuse texture
    std::string baseColorFilePath;

    // Occlusion Roughness Metallic texture
    std::string ormFilePath;

    // Normal texture
    std::string normalFilePath;

    Glpa::FileDataManager* fileDataManager = nullptr;

public :
    Material(std::string argName, std::string argBaseColorFilePath);
    ~Material();

    bool isLoaded() const {return loaded;}

    void setManager(Glpa::FileDataManager* argFileDataManager);

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    int GetMtWidth(std::string mtName);
    int GetMtHeight(std::string mtName);
    LPDWORD GetMtData(std::string mtName);

    Glpa::GPU_MATERIAL getData();

    void load();
    void release();



};


}



#endif GLPA_MATERIAL_H_