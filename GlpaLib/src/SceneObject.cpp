#include "SceneObject.h"
#include "GlpaLog.h"

Glpa::SceneObject::SceneObject()
{
    Glpa::OutputLog("__FILE__", __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
    type = Glpa::CLASS_SCENE_OBJECT;
}

Glpa::SceneObject::~SceneObject()
{
    Glpa::OutputLog("__FILE__", __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}
