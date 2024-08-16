#ifndef GLPA_STATIONARY_OBJECT_H_
#define GLPA_STATIONARY_OBJECT_H_

#include <string>

#include "SceneObject.h"
#include "Vector.h"

namespace Glpa
{

class StationaryObject : public Glpa::SceneObject
{
protected :
    bool visible = true;
    Glpa::Vec3d pos;
    Glpa::Vec3d rotate;
    Glpa::Vec3d scale;

public :
    StationaryObject(std::string argName, std::string argFilePath, Glpa::Vec3d defPos);
    ~StationaryObject() override;

    bool GetVisible() const {return visible;}
    void SetVisible(bool symbol) {visible = symbol;}

    Glpa::Vec3d GetPos() const {return pos;}
    void SetPos(Glpa::Vec3d value) {pos = value;}

    Glpa::Vec3d GetRotate() const {return rotate;}
    void SetRotate(Glpa::Vec3d value) {rotate = value;}

    Glpa::Vec3d GetScale() const {return scale;}
    void SetScale(Glpa::Vec3d value) {scale = value;}

    virtual void load() = 0;
    virtual void release() = 0;


};



}


#endif GLPA_STATIONARY_OBJECT_H_