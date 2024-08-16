#include "Material.h"
#include "GlpaLog.h"

Glpa::Material::Material(std::string argName, std::string argDiffuseFilePath)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    diffuseFilePath = argDiffuseFilePath;
    
}

void Glpa::Material::setManager(Glpa::FileDataManager *argFileDataManager)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    fileDataManager = argFileDataManager;
}
