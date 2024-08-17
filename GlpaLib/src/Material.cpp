#include "Material.h"
#include "ErrorHandler.h"

Glpa::Material::Material(std::string argName, std::string argDiffuseFilePath)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    diffuseFilePath = argDiffuseFilePath;
    
}

Glpa::Material::~Material()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Material::setManager(Glpa::FileDataManager *argFileDataManager)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    fileDataManager = argFileDataManager;
}

int Glpa::Material::GetMtWidth(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getWidth(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getWidth(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getWidth(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return -1;
    };
}

int Glpa::Material::GetMtHeight(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getHeight(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getHeight(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getHeight(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return -1;
    };
}

LPDWORD Glpa::Material::GetMtData(std::string mtName)
{
    if (mtName == Glpa::MATERIAL_DIFFUSE) return fileDataManager->getData(diffuseFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getData(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getData(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return LPDWORD();
    };
}

void Glpa::Material::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = true;

    fileDataManager->newFile(diffuseFilePath);
    fileDataManager->load(diffuseFilePath);
}

void Glpa::Material::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = false;

    fileDataManager->release(diffuseFilePath);
}
