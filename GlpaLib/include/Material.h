#ifndef GLPA_MATERIAL_H_
#define GLPA_MATERIAL_H_

#include <string>

#include "FileDataManager.h"

namespace Glpa
{

class Material
{
private :
    bool loaded = false;
    std::string name;

    std::string diffuseFilePath;
    std::string aoFilePath;
    std::string roughnessFilePath;
    std::string normalFilePath;

    Glpa::FileDataManager* fileDataManager = nullptr;

public :
    Material(std::string argName, std::string argDiffuseFilePath);
    ~Material();

    bool isLoaded() const {return loaded;}

    void setManager(Glpa::FileDataManager* argFileDataManager);

    std::string getName() const {return name;}
    void setName(std::string str) {name = str;}

    void load();
    void release();

};


}



#endif GLPA_MATERIAL_H_