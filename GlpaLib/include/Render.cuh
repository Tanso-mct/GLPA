#ifndef GLPA_RENDER_CU_H_
#define GLPA_RENDER_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "SceneObject.h"

#include "Image.h"
#include "Color.h"
#include "Material.cuh"
#include "Camera.cuh"
#include "StationaryObject.cuh"
#include "GpuData.cuh"

#include <unordered_map>
#include <map>
#include <Window.h>

#include <algorithm>

namespace Glpa
{

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


typedef struct _GPU_RENDER_RESULT
{
    GPU_BOOL onErr = FALSE;
    int srcObjSum;
    int objSum;
    int polySum;

    int facingPolySum;

    int facingPolyI[212];
    int facingObjI[212];

    int insidePolySum;
    int insidePolyRangeSum;

    int needClipPolySum;

    int polyFaceInxtnSum;
    int vvFaceInxtnSum;

    int inxtnObjId[212];
    int inxtnPolyId[212];
    int inxtnAmountsPoly[212];
    int inxtnAmountsVv[212];

    float mPolyCubeVs[12][7][3];
    float mPolyPlaneVs[200][7][3];

    int maxLineAmount;
    int maxPolyHeight;
    int debugNum;

    int rasterizePolySum;
} GPU_RENDER_RESULT;

typedef struct _GPU_Z_BUFFER_ARRAY
{
    int isEmpt = 0;
    int objId = 0;
    int polyId = 0;

    Glpa::GPU_VEC_3D v;
    Glpa::GPU_VEC_2D scrV;
} GPU_Z_BUFFER_ARY;

class RENDER_RESULT_FACTORY
{
public :
    int* hPolyAmounts = nullptr;
    Glpa::GPU_RENDER_RESULT hResult;
    bool malloced = false;

    RENDER_RESULT_FACTORY()
    {
        malloced = false;
    }

    void dFree(Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts);
    void dMalloc
    (
        Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts, int srcObjSum, 
        int bufWidth, int bufHeight, int bufDpi
    );

    void deviceToHost(Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts);
};

class Render3d
{
private :
    Glpa::CAMERA_FACTORY camFactory;
    Glpa::GPU_CAMERA* dCamData = nullptr;

    Glpa::MATERIAL_FACTORY mtFactory;
    Glpa::GPU_MATERIAL* dMts = nullptr;

    Glpa::ST_OBJECT_FACTORY stObjFactory;
    Glpa::GPU_ST_OBJECT_DATA* dStObjData = nullptr;
    Glpa::GPU_POLYGON** dObjPolys;
    Glpa::GPU_ST_OBJECT_INFO* dStObjInfo = nullptr;

    GPU_MPOLYGON_INFO* dMPolyInfo = nullptr;
    Glpa::GPU_POLY_LINE** dPolyLines = nullptr;

    Glpa::RENDER_RESULT_FACTORY resultFactory;
    Glpa::GPU_RENDER_RESULT* dResult = nullptr;
    int* dPolyAmounts = nullptr;

    LPDWORD dBuf = nullptr;

    void dMalloc
    (
        Glpa::Camera& cam, 
        std::unordered_map<std::string, Glpa::SceneObject*>& objs, 
        std::unordered_map<std::string, Glpa::Material*>& mts,
        int& bufWidth, int& bufHeight, int& bufDpi
    );

    void prepareObjs();
    void prepareLines();
    void rasterize(int& bufWidth, int& bufHeight, int& bufDpi, LPDWORD buf);

public :
    Render3d();
    ~Render3d();

    void run
    (
        std::unordered_map<std::string, Glpa::SceneObject*>& objs, 
        std::unordered_map<std::string, Glpa::Material*>& mts, Glpa::Camera& cam,
        LPDWORD buf, int& bufWidth, int& bufHeight, int& bufDpi
    );

    void dRelease();
};

}

#endif GLPA_RENDER_CU_H_