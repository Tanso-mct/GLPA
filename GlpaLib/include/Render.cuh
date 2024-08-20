#ifndef GLPA_RENDER_CU_H_
#define GLPA_RENDER_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SceneObject.h"

#include "Image.h"
#include "Color.h"
#include "Material.h"
#include "Camera.h"
#include "StationaryObject.h"

#include <unordered_map>
#include <map>
#include <Window.h>

#include <algorithm>

namespace Glpa
{

__global__ void Gpu2dDraw
(
    int* imgPosX,
    int* imgPosY,
    int* imgWidth,
    int* imgHeight,
    LPDWORD* imgData,
    int imgAmount,
    LPDWORD buf,
    int bufWidth,
    int bufHeight,
    int bufDpi,
    DWORD background
);

class Render2d
{
private :
    bool malloced = false;

    std::vector<std::string> imgNames;
    std::vector<int> hImgPosX;
    std::vector<int> hImgPosY;
    std::vector<int> hImgWidth;
    std::vector<int> hImgHeight;
    std::vector<LPDWORD> hImgData;

    LPDWORD dBuf = nullptr;

    int* dImgPosX;
    int* dImgPosY;
    int* dImgWidth;
    int* dImgHeight;
    int* dImgDrawOrder;
    LPDWORD* dImgData;

    int imgAmount = 0;

    int maxImgWidth = 0;
    int maxImgHeight = 0;

    DWORD backgroundColor;

public :
    Render2d();
    ~Render2d();

    void setBackground(std::string color, DWORD& bg);

    void editObjsPos(Glpa::Image *img);
    void editBufSize(int bufWidth, int bufHeight, int bufDpi);

    void dMalloc
    (
        std::unordered_map<std::string, Glpa::SceneObject*>& objs,
        std::map<int, std::vector<std::string>>& drawOrderMap, std::vector<std::string>& drawOrder,
        int bufWidth, int bufHeight, int bufDpi, std::string bgColor
    );
    void dRelease();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*>& objs,
        std::map<int, std::vector<std::string>>& drawOrderMap, std::vector<std::string>& drawOrder,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
    );
};


__global__ void GpuPrepareObj
(
    Glpa::OBJECT3D_DATA* objData,
    Glpa::OBJECT_INFO* objInfo,
    Glpa::CAMERA* camData,
    int objAmount    
);

class Render3d
{
private :
    bool camMalloced = false;
    bool objMtDataMalloced = false;
    bool objInfoMalloced = false;

    std::unordered_map<std::string, int> mtIdMap;
    std::unordered_map<std::string, int> objIdMap;

    Glpa::CAMERA* dCamData;
    Glpa::MATERIAL* dMts;
    Glpa::OBJECT3D_DATA* dObjData;
    Glpa::OBJECT_INFO* dObjInfo;

    void dMallocCam(Glpa::Camera& cam);
    void dReleaseCam();

    void dMallocObjsMtData
    (
        std::unordered_map<std::string, Glpa::SceneObject*>& objs, 
        std::unordered_map<std::string, Glpa::Material*>& mts
    );
    void dReleaseObjsMtData();

    void dMallocObjInfo(std::unordered_map<std::string, Glpa::SceneObject*>& objs);
    void dReleaseObjInfo();

    void prepareObjs();
    void setVs();
    void rasterize();

public :
    Render3d();
    ~Render3d();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*>& objs, 
        std::unordered_map<std::string, Glpa::Material*>& mts, Glpa::Camera& cam,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
    );

    void dRelease();
};

}

#endif GLPA_RENDER_CU_H_