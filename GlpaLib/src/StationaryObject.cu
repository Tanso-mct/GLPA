#include "StationaryObject.cuh"
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

Glpa::GPU_ST_OBJECT_INFO Glpa::StationaryObject::getInfo()
{
    Glpa::GPU_ST_OBJECT_INFO info;

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

void Glpa::ST_OBJECT_FACTORY::dFree(Glpa::GPU_ST_OBJECT_DATA*& dObjData)
{
    if (dataMalloced)
    {
        for (size_t i = 0; i < idMap.size(); i++)
        {
            Glpa::GPU_POLYGON* dPolygons = nullptr;
            cudaMemcpy(dPolygons, &dObjData[i].polygons, sizeof(Glpa::GPU_ST_OBJECT_DATA*), cudaMemcpyDeviceToHost);
            cudaFree(dPolygons);
        };

        free(dObjData);
        dObjData = nullptr;

        dataMalloced = false;
    };
}

void Glpa::ST_OBJECT_FACTORY::dFree(Glpa::GPU_ST_OBJECT_INFO*& dObjInfo)
{
    if (infoMalloced)
    {
        free(dObjInfo);
    };
}

void Glpa::ST_OBJECT_FACTORY::dMalloc
(
    Glpa::GPU_ST_OBJECT_DATA*& dObjData, 
    std::unordered_map<std::string, Glpa::SceneObject *> &sObjs, 
    std::unordered_map<std::string, int> &mtIdMap
){
    if (dataMalloced) dFree(dObjData);

    std::vector<Glpa::GPU_ST_OBJECT_DATA> hObjs;
    int id = 0;
    for (auto& pair : sObjs)
    {
        if (Glpa::StationaryObject* obj = dynamic_cast<Glpa::StationaryObject*>(pair.second))
        {
            Glpa::GPU_ST_OBJECT_DATA objData;
            objData.id = id;
            objData.mtId = mtIdMap[obj->GetMaterial()->getName()];

            std::vector<Glpa::GPU_POLYGON> polygons = obj->getPolyData();
            objData.polyAmount = polygons.size();

            GPU_POLYGON* dPolygons = nullptr;
            cudaMalloc(&dPolygons, polygons.size() * sizeof(Glpa::GPU_POLYGON));
            cudaMemcpy(dPolygons, polygons.data(), polygons.size() * sizeof(Glpa::GPU_POLYGON), cudaMemcpyHostToDevice);

            cudaMalloc(&objData.polygons, polygons.size() * sizeof(Glpa::GPU_POLYGON));
            cudaMemcpy(objData.polygons, dPolygons, polygons.size() * sizeof(Glpa::GPU_POLYGON), cudaMemcpyDeviceToDevice);

            cudaFree(dPolygons);

            objData.range = obj->getRangeRectData();

            idMap[pair.first] = id;

            hObjs.push_back(objData);
            id++;
        }
    }

    cudaMalloc(&dObjData, id * sizeof(Glpa::GPU_ST_OBJECT_DATA));
    cudaMemcpy(dObjData, hObjs.data(), id * sizeof(Glpa::GPU_ST_OBJECT_DATA), cudaMemcpyHostToDevice);

    for (int i = 0; i < hObjs.size(); i++)
    {
        cudaFree(hObjs[i].polygons);
    }

    dataMalloced = true;
}

void Glpa::ST_OBJECT_FACTORY::dMalloc
(
    Glpa::GPU_ST_OBJECT_INFO*& dObjInfo, 
    std::unordered_map<std::string, Glpa::SceneObject *> &sObjs
){
    std::vector<Glpa::GPU_ST_OBJECT_INFO> hObjInfo;
    for (auto& pair : idMap)
    {
        if (Glpa::StationaryObject* obj = dynamic_cast<Glpa::StationaryObject*>(sObjs[pair.first]))
        {
            hObjInfo.push_back(obj->getInfo());
        }
    }

    cudaMalloc(&dObjInfo, idMap.size() * sizeof(Glpa::GPU_ST_OBJECT_INFO));
    cudaMemcpy(dObjInfo, hObjInfo.data(), idMap.size() * sizeof(Glpa::GPU_ST_OBJECT_INFO), cudaMemcpyHostToDevice);
}
