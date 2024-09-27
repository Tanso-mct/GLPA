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

int Glpa::StationaryObject::getPolyAmount()
{
    return fileDataManager->getPolyAmount(filePath);
}

void Glpa::StationaryObject::getPolyData(std::vector<Glpa::GPU_POLYGON> &polys)
{
    fileDataManager->getPolyData(filePath, polys);
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

void Glpa::ST_OBJECT_FACTORY::dFree(Glpa::GPU_ST_OBJECT_DATA*& dObjData, Glpa::GPU_POLYGON**& dPolys)
{
    if (!dataMalloced) return;

    cudaFree(dObjData);
    dObjData = nullptr;

    for (int i = 0; i < idMap.size(); i++)
    {
        cudaFree(dPolys[i]);
    }

    cudaFree(dPolys);
    dPolys = nullptr;

    dataMalloced = false;
}

void Glpa::ST_OBJECT_FACTORY::dFree(Glpa::GPU_ST_OBJECT_INFO*& dObjInfo)
{
    if (!infoMalloced) return;

    cudaFree(dObjInfo);
    infoMalloced = false;
}

__global__ void freeInKanel(Glpa::GPU_POLY_LINE** dPolyLine, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(dPolyLine[i]);
    }
}

void Glpa::ST_OBJECT_FACTORY::dFree(Glpa::GPU_POLY_LINE **&dPolyLines, Glpa::GPU_MPOLYGON_INFO*& dMPolyInfo)
{
    if (!polyLineMalloced) return;

    freeInKanel<<<1, 1>>>(dPolyLines, lastPolySum);
    cudaDeviceSynchronize();
    checkCudaErr(__FILE__, __LINE__, __FUNCSIG__);

    cudaFree(dPolyLines);
    dPolyLines = nullptr;

    cudaFree(dMPolyInfo);

    polyLineMalloced = false;
}

void Glpa::ST_OBJECT_FACTORY::dMalloc
(
    Glpa::GPU_ST_OBJECT_DATA*& dObjData, 
    Glpa::GPU_POLYGON**& dPolys,
    std::unordered_map<std::string, Glpa::SceneObject *> &sObjs, 
    std::unordered_map<std::string, int> &mtIdMap
){
    if (dataMalloced) return;

    std::vector<Glpa::GPU_ST_OBJECT_DATA> hObjs;
    std::vector<Glpa::GPU_POLYGON*> polys;
    int id = 0;
    for (auto& pair : sObjs)
    {
        if (Glpa::StationaryObject* obj = dynamic_cast<Glpa::StationaryObject*>(pair.second))
        {
            Glpa::GPU_ST_OBJECT_DATA objData;
            objData.id = id;
            objData.mtId = mtIdMap[obj->GetMaterial()->getName()];

            std::vector<Glpa::GPU_POLYGON> hThisPolys;
            obj->getPolyData(hThisPolys);

            Glpa::GPU_POLYGON* dThisPolys;
            cudaMalloc(&dThisPolys, obj->getPolyAmount() * sizeof(Glpa::GPU_POLYGON));
            cudaMemcpy(dThisPolys, hThisPolys.data(), obj->getPolyAmount() * sizeof(Glpa::GPU_POLYGON), cudaMemcpyHostToDevice);
            polys.push_back(dThisPolys);

            objData.polyAmount = obj->getPolyAmount();
            objData.range = obj->getRangeRectData();
            idMap[pair.first] = id;

            hObjs.push_back(objData);
            id++;
        }
    }

    cudaMalloc(&dObjData, id * sizeof(Glpa::GPU_ST_OBJECT_DATA));
    cudaMemcpy(dObjData, hObjs.data(), id * sizeof(Glpa::GPU_ST_OBJECT_DATA), cudaMemcpyHostToDevice);

    cudaMalloc(&dPolys, polys.size() * sizeof(Glpa::GPU_POLYGON*));
    cudaMemcpy(dPolys, polys.data(), polys.size() * sizeof(Glpa::GPU_POLYGON*), cudaMemcpyHostToDevice);

    dataMalloced = true;
}

void Glpa::ST_OBJECT_FACTORY::dMalloc
(
    Glpa::GPU_ST_OBJECT_INFO*& dObjInfo, 
    std::unordered_map<std::string, Glpa::SceneObject *> &sObjs
){
    if (infoMalloced) return;

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

    infoMalloced = true;
}

void Glpa::ST_OBJECT_FACTORY::dMalloc(Glpa::GPU_POLY_LINE**& dPolyLines, GPU_MPOLYGON_INFO*& dMPolyInfo, int polySum)
{
    if (polyLineMalloced) return;

    cudaMalloc(&dPolyLines, polySum * sizeof(Glpa::GPU_POLY_LINE*));
    cudaMalloc(&dMPolyInfo, polySum * sizeof(GPU_MPOLYGON_INFO));

    lastPolySum = polySum;

    polyLineMalloced = true;
}
