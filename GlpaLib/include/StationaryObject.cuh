#ifndef GLPA_STATIONARY_OBJECT_H_
#define GLPA_STATIONARY_OBJECT_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>

#include "SceneObject.h"
#include "Material.cuh"

#include "Vector.cuh"

namespace Glpa
{

typedef struct _GPU_STATIONARY_OBJECT_DATA
{
    int id;
    int mtId;
    int polyAmount;
    Glpa::GPU_RANGE_RECT range;
} GPU_ST_OBJECT_DATA;

typedef struct _GPU_STATIONARY_OBJECT_INFO
{
    GPU_BOOL isVisible = TRUE;
    Glpa::GPU_VEC_3D pos;
    Glpa::GPU_VEC_3D rot;
    Glpa::GPU_VEC_3D scale;

    GPU_BOOL isInVV = FALSE;
} GPU_ST_OBJECT_INFO;

class StationaryObject : public Glpa::SceneObject
{
protected :
    bool visible = true;
    Glpa::Vec3d pos;
    Glpa::Vec3d rotate;
    Glpa::Vec3d scale;

    Glpa::Material* mt = nullptr;

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

    void SetMaterial(Glpa::Material* value) {mt = value;}
    Glpa::Material* GetMaterial() const {return mt;}

    Glpa::GPU_ST_OBJECT_INFO getInfo();

    void getPolyData(std::vector<Glpa::GPU_POLYGON>& polys);
    Glpa::GPU_RANGE_RECT getRangeRectData();

    void load();
    void release();
};

class ST_OBJECT_FACTORY
{
public :
    std::unordered_map<std::string, int> idMap;

    bool dataMalloced = false;
    bool infoMalloced = false;

    ST_OBJECT_FACTORY()
    {
        dataMalloced = false;
        infoMalloced = false;
    }

    void dFree(Glpa::GPU_ST_OBJECT_DATA*& dObjData, Glpa::GPU_POLYGON**& dPolys);
    void dFree(Glpa::GPU_ST_OBJECT_INFO*& dObjInfo);

    void dMalloc
    (
        Glpa::GPU_ST_OBJECT_DATA*& dObjData,
        Glpa::GPU_POLYGON**& dPolys,
        std::unordered_map<std::string, Glpa::SceneObject*>& sObjs,
        std::unordered_map<std::string, int>& mtIdMap
    );

    void dMalloc
    (
        Glpa::GPU_ST_OBJECT_INFO*& dObjInfo, 
        std::unordered_map<std::string, Glpa::SceneObject*>& sObjs
    );
};



}


#endif GLPA_STATIONARY_OBJECT_H_