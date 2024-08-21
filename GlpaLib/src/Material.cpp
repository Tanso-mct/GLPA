#include "Material.h"
#include "ErrorHandler.h"

Glpa::Material::Material(std::string argName, std::string argBaseColorFilePath)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    baseColorFilePath = argBaseColorFilePath;
    
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
    if (mtName == Glpa::MATERIAL_BASE_COLOR) return fileDataManager->getWidth(baseColorFilePath);
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
    if (mtName == Glpa::MATERIAL_BASE_COLOR) return fileDataManager->getHeight(baseColorFilePath);
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
    if (mtName == Glpa::MATERIAL_BASE_COLOR) return fileDataManager->getPngData(baseColorFilePath);
    else if (mtName == Glpa::MATERIAL_ORM) return fileDataManager->getPngData(ormFilePath);
    else if (mtName == Glpa::MATERIAL_NORMAL) return fileDataManager->getPngData(normalFilePath);
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, {mtName + " is not a valid material name"});
        return LPDWORD();
    };
}

Glpa::GPU_MATERIAL Glpa::Material::getData()
{
    Glpa::GPU_MATERIAL material;

    material.baseColor = fileDataManager->getPngData(baseColorFilePath);

    return material;
}

void Glpa::Material::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = true;

    fileDataManager->newFile(baseColorFilePath);
    fileDataManager->load(baseColorFilePath);
}

void Glpa::Material::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Material[" + name + "]");
    loaded = false;

    fileDataManager->release(baseColorFilePath);
}
