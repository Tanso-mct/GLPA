#include "StationaryObject.h"
#include "GlpaLog.h"

Glpa::StationaryObject::StationaryObject(std::string argName, std::string argFilePath, Glpa::Vec3d defPos)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    name = argName;
    filePath = argFilePath;
    type = Glpa::CLASS_STATIONARY_OBJECT;
}

Glpa::StationaryObject::~StationaryObject()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::StationaryObject::load()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "StationaryObject[" + name + "]");
    loaded = true;

    fileDataManager->newFile(filePath);
    fileDataManager->load(filePath);
}

void Glpa::StationaryObject::release()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "StationaryObject[" + name + "]");
    loaded = false;

    fileDataManager->release(filePath);
}