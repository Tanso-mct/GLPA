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

Glpa::GPU_OBJECT3D_INFO Glpa::StationaryObject::getInfo()
{
    Glpa::GPU_OBJECT3D_INFO info;

    info.isVisible = (visible) ? TRUE : FALSE;
    info.pos.x = pos.x;
    info.pos.y = pos.y;
    info.pos.z = pos.z;

    info.rot.x = rotate.x;
    info.rot.y = rotate.y;
    info.rot.z = rotate.z;

    info.scale.x = scale.x;
    info.scale.y = scale.y;
    info.scale.z = scale.z;

    return info;
}

std::vector<Glpa::GPU_POLYGON> Glpa::StationaryObject::getPolyData()
{
    return fileDataManager->getPolyData(filePath);
}

Glpa::GPU_RANGE_RECT Glpa::StationaryObject::getRangeRectData()
{
    return fileDataManager->getRangeRectData(filePath);
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