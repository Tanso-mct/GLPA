#include "SceneObject.h"
#include "GlpaLog.h"

Glpa::SceneObject::SceneObject()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    type = Glpa::CLASS_SCENE_OBJECT;
}

Glpa::SceneObject::~SceneObject()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::SceneObject::setManager(Glpa::FileDataManager *argFileDataManager)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Object[" + name + "]");
    fileDataManager = argFileDataManager;
}
