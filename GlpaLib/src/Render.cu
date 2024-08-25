#include "Render.cuh"

#include "GlpaLog.h"
#include "GlpaConsole.h"

Glpa::Render2d::Render2d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::Render2d::~Render2d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Render2d::setBackground(std::string color, DWORD& bg)
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_RENDER, color.c_str());
    if (color == Glpa::COLOR_BLACK)
    {
        Glpa::Color instColor(0, 0, 0, 1);
        bg = instColor.GetDword();
    }
    else if (color == Glpa::COLOR_GREEN)
    {
        Glpa::Color instColor(0, 200, 0, 1);
        bg = instColor.GetDword();
    }
    else
    {
        Glpa::Color instColor(0, 200, 0, 1);
        bg = instColor.GetDword();
    }
}

void Glpa::Render2d::editObjsPos(Glpa::Image *img){
    if (!malloc) return;

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_RENDER, img->getName().c_str());
    cudaFree(dImgPosX);
    cudaFree(dImgPosY);

    int index = std::distance(imgNames.begin(), std::find(imgNames.begin(), imgNames.end(), img->getName()));

    Vec2d imgPos = img->GetPos();
    hImgPosX[index] = imgPos.x;
    hImgPosY[index] = imgPos.y;

    cudaMalloc(&dImgPosX, hImgPosX.size() * sizeof(int));
    cudaMemcpy(dImgPosX, hImgPosX.data(), hImgPosX.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dImgPosY, hImgPosY.size() * sizeof(int));
    cudaMemcpy(dImgPosY, hImgPosY.data(), hImgPosY.size() * sizeof(int), cudaMemcpyHostToDevice);

}

void Glpa::Render2d::editBufSize(int bufWidth, int bufHeight, int bufDpi)
{
    if (!malloc) return;

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_RENDER, "");
    cudaFree(dBuf);
    cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
}

void Glpa::Render2d::dMalloc
(
    std::unordered_map<std::string, Glpa::SceneObject*>& objs,
    std::map<int, std::vector<std::string>>& drawOrderMap, std::vector<std::string>& drawOrder,
    int bufWidth, int bufHeight, int bufDpi, std::string bgColor
){
    if (malloced) return;

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_RENDER, "");

    maxImgWidth = 0;
    maxImgHeight = 0;
    
    hImgPosX.clear();
    hImgPosY.clear();
    hImgWidth.clear();
    hImgHeight.clear();

    imgNames.clear();
    drawOrder.clear();

    hImgData.clear();
    for (int i = 0; i < hImgData.size(); i++)
    {
        cudaFree(hImgData[i]);
    }

    for (auto& pair : drawOrderMap)
    {
        for (int i = 0; i < pair.second.size(); i++)
        {
            if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[pair.second[i]]))
            {
                if (img->getVisible())
                {
                    Vec2d imgPos = img->GetPos();
                    hImgPosX.push_back(imgPos.x);
                    hImgPosY.push_back(imgPos.y);
                    hImgWidth.push_back(img->GetWidth());
                    hImgHeight.push_back(img->GetHeight());

                    imgNames.push_back(img->getName());
                    drawOrder.push_back(img->getName());

                    maxImgWidth = (maxImgWidth < img->GetWidth()) ? img->GetWidth() : maxImgWidth;
                    maxImgHeight = (maxImgHeight < img->GetHeight()) ? img->GetHeight() : maxImgHeight;

                    LPDWORD dData;
                    int dataSize = img->GetWidth() * img->GetHeight() * sizeof(DWORD);
                    cudaMalloc(&dData, dataSize);
                    cudaMemcpy(dData, img->GetData(), dataSize, cudaMemcpyHostToDevice);
                    hImgData.push_back(dData);
                }
            }
        }
    }

    imgAmount = hImgData.size();

    setBackground(bgColor, backgroundColor);

    cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));

    cudaMalloc(&dImgPosX, hImgPosX.size() * sizeof(int));
    cudaMemcpy(dImgPosX, hImgPosX.data(), hImgPosX.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dImgPosY, hImgPosY.size() * sizeof(int));
    cudaMemcpy(dImgPosY, hImgPosY.data(), hImgPosY.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dImgWidth, hImgWidth.size() * sizeof(int));
    cudaMemcpy(dImgWidth, hImgWidth.data(), hImgWidth.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dImgHeight, hImgHeight.size() * sizeof(int));
    cudaMemcpy(dImgHeight, hImgHeight.data(), hImgHeight.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dImgData, hImgData.size() * sizeof(DWORD*));
    cudaMemcpy(dImgData, hImgData.data(), hImgData.size() * sizeof(DWORD*), cudaMemcpyHostToDevice);

    malloced = true;

}

void Glpa::Render2d::dRelease()
{
    if (!malloced) return;

    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_RENDER, "");

    cudaFree(dBuf);
    cudaFree(dImgPosX);
    cudaFree(dImgPosY);
    cudaFree(dImgWidth);
    cudaFree(dImgHeight);

    for (int i = 0; i < hImgData.size(); i++)
    {
        LPDWORD ptDeviceData;
    
        cudaMemcpy(&ptDeviceData, &dImgData[i], sizeof(LPDWORD), cudaMemcpyDeviceToHost);
        cudaFree(ptDeviceData);
    }

    cudaFree(dImgData);

    malloced = false;
}

__global__ void Gpu2dDraw
(
    int *imgPosX, int *imgPosY, int* imgWidth, int* imgHeight, LPDWORD *imgData, int imgAmount,
    LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, DWORD background
){
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < bufWidth)
    {
        if (y < bufHeight)
        {
            atomicExch((unsigned int*)&buf[x + y * bufWidth * bufDpi], (unsigned int)background);

            int isBackgroundIF = TRUE;
            int isNotBackgroundIF = FALSE;
            for (int i = 0; i < imgAmount; i++)
            {
                int xInImgIF = (x >= imgPosX[i] && x < imgPosX[i] + imgWidth[i]) ? TRUE : FALSE;
                int yInImgIF = (y >= imgPosY[i] && y < imgPosY[i] + imgHeight[i]) ? TRUE : FALSE;

                int writeIF = (xInImgIF == TRUE && yInImgIF == TRUE) ? TRUE : FALSE;

                for (int cb1 = 0; cb1 < writeIF; cb1++)
                {
                    int imgX = x - imgPosX[i];
                    int imgY = y - imgPosY[i];

                    for (int cb2 = 0; cb2 < isNotBackgroundIF; cb2++)
                    {
                        BYTE bufR = (buf[x + y * bufWidth * bufDpi] >> 16) & 0xFF;
                        BYTE bufG = (buf[x + y * bufWidth * bufDpi] >> 8) & 0xFF;
                        BYTE bufB = buf[x + y * bufWidth * bufDpi] & 0xFF;

                        BYTE imgA = (imgData[i][imgX + imgY * imgWidth[i]] >> 24) & 0xFF;
                        BYTE imgR = (imgData[i][imgX + imgY * imgWidth[i]] >> 16) & 0xFF;
                        BYTE imgG = (imgData[i][imgX + imgY * imgWidth[i]] >> 8) & 0xFF;
                        BYTE imgB = imgData[i][imgX + imgY * imgWidth[i]] & 0xFF;

                        float alpha = static_cast<float>(imgA) / 255.0f;
                        float invAlpha = 1.0f - alpha;

                        bufR = static_cast<unsigned char>(alpha * imgR + invAlpha * bufR);
                        bufG = static_cast<unsigned char>(alpha * imgG + invAlpha * bufG);
                        bufB = static_cast<unsigned char>(alpha * imgB + invAlpha * bufB);

                        DWORD newColor = (1 << 24) | (bufR << 16) | (bufG << 8) | bufB;
                        atomicExch((unsigned int*)&buf[x + y * bufWidth * bufDpi], (unsigned int)newColor);
                    }

                    for (int cb2 = 0; cb2 < isBackgroundIF; cb2++)
                    {
                        atomicExch((unsigned int*)&buf[x + y * bufWidth * bufDpi], (unsigned int)imgData[i][imgX + imgY * imgWidth[i]]);
                        isBackgroundIF = FALSE;
                        isNotBackgroundIF = TRUE;
                    }

                }
            }
        }
    }
}

void Glpa::Render2d::run
(
        std::unordered_map<std::string, Glpa::SceneObject*>& objs,
        std::map<int, std::vector<std::string>>& drawOrderMap, std::vector<std::string>& drawOrder,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
){
    dMalloc(objs, drawOrderMap, drawOrder, bufWidth, bufHeight, bufDpi, bgColor);

    if (imgAmount != 0)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = bufWidth;
        int dataSizeX = bufHeight;

        int desiredThreadsPerBlockX = 16;
        int desiredThreadsPerBlockY = 16;

        int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
        int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

        blocksX = min(blocksX, deviceProp.maxGridSize[0]);
        blocksY = min(blocksY, deviceProp.maxGridSize[1]);

        int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
        int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

        dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
        dim3 dimGrid(blocksX, blocksY);

        Gpu2dDraw<<<dimGrid, dimBlock>>>
        (
            dImgPosX, dImgPosY, dImgWidth, dImgHeight, dImgData, imgAmount, 
            dBuf, bufWidth, bufHeight, bufDpi, backgroundColor
        );
        cudaError_t error = cudaDeviceSynchronize();
        if (error != 0){
            Glpa::runTimeError(__FILE__, __LINE__, {"Processing with Cuda failed."});
        }

        cudaMemcpy(buf, dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(buf, dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyDeviceToHost);
    }
}

Glpa::Render3d::Render3d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Constructor");
}

Glpa::Render3d::~Render3d()
{
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "Destructor");
}

void Glpa::Render3d::dMallocCam(Glpa::Camera& cam)
{
    if (camMalloced) dReleaseCam();
    cudaError_t err;

    err = cudaMalloc(&dCamData, sizeof(Glpa::GPU_CAMERA));
    err = cudaMemcpy(dCamData, &cam.getData(), sizeof(Glpa::GPU_CAMERA), cudaMemcpyHostToDevice);

    camMalloced = true;
}

void Glpa::Render3d::dReleaseCam()
{
    if (!camMalloced) return;
    cudaError_t err;

    err = cudaFree(dCamData);

    camMalloced = false;
}

void Glpa::Render3d::dMallocObjsMtData
(
    std::unordered_map<std::string, Glpa::SceneObject*>& objs,
    std::unordered_map<std::string, Glpa::Material*>& mts
){
    if (objMtDataMalloced) return;
    cudaError_t err;
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "");

    // Material data
    std::vector<Glpa::GPU_MATERIAL> hMts;
    int mtId = 0;
    for (auto& pair : mts)
    {
        Glpa::GPU_MATERIAL mt;
        err = cudaMalloc
        (
            &mt.baseColor, 
            pair.second->GetMtWidth(Glpa::MATERIAL_DIFFUSE) * pair.second->GetMtHeight(Glpa::MATERIAL_DIFFUSE) * sizeof(DWORD)
        );
        err = cudaMemcpy
        (
            mt.baseColor, 
            pair.second->GetMtData(Glpa::MATERIAL_DIFFUSE), 
            pair.second->GetMtWidth(Glpa::MATERIAL_DIFFUSE) * pair.second->GetMtHeight(Glpa::MATERIAL_DIFFUSE) * sizeof(DWORD), 
            cudaMemcpyHostToDevice
        );

        mtIdMap[pair.first] = mtId;

        hMts.push_back(mt);
        mtId++;
    }
    err = cudaMalloc(&dMts, mtId * sizeof(Glpa::GPU_MATERIAL));
    err = cudaMemcpy(dMts, hMts.data(), mtId * sizeof(Glpa::GPU_MATERIAL), cudaMemcpyHostToDevice);

    for (int i = 0; i < hMts.size(); i++)
    {
        err = cudaFree(hMts[i].baseColor);
    }

    // Object data
    std::vector<Glpa::GPU_OBJECT3D_DATA> hObjData;
    int objId = 0;
    for (auto& pair : objs)
    {
        if (Glpa::StationaryObject* obj = dynamic_cast<Glpa::StationaryObject*>(pair.second))
        {
            Glpa::GPU_OBJECT3D_DATA objData;
            objData.id = objId;
            objData.mtId = mtIdMap[obj->GetMaterial()->getName()];

            std::vector<Glpa::GPU_POLYGON> polygons = obj->getPolyData();
            err = cudaMalloc(&objData.polygons, polygons.size() * sizeof(Glpa::GPU_POLYGON));
            err = cudaMemcpy(objData.polygons, polygons.data(), polygons.size() * sizeof(Glpa::GPU_POLYGON), cudaMemcpyHostToDevice);

            objData.range = obj->getRangeRectData();

            objIdMap[pair.first] = objId;

            hObjData.push_back(objData);
            objId++;
        }
    }

    err = cudaMalloc(&dObjData, objId * sizeof(Glpa::GPU_OBJECT3D_DATA));
    err = cudaMemcpy(dObjData, hObjData.data(), objId * sizeof(Glpa::GPU_OBJECT3D_DATA), cudaMemcpyHostToDevice);

    for (int i = 0; i < hObjData.size(); i++)
    {
        err = cudaFree(hObjData[i].polygons);
    }

    objMtDataMalloced = true;
}

void Glpa::Render3d::dReleaseObjsMtData()
{
    if (!objMtDataMalloced) return;
    cudaError_t err;
    Glpa::OutputLog(__FILE__, __LINE__, __FUNCSIG__, Glpa::OUTPUT_TAG_GLPA_LIB, "");

    for (int i = 0; i < mtIdMap.size(); i++)
    {
        LPDWORD dBaseColor = nullptr;
        err = cudaMemcpy(dBaseColor, &dMts[i].baseColor, sizeof(LPDWORD), cudaMemcpyDeviceToHost);
        err = cudaFree(dBaseColor);
    }
    err = cudaFree(dMts);

    for (int i = 0; i < objIdMap.size(); i++)
    {
        Glpa::GPU_POLYGON* dPolygons = nullptr;
        err = cudaMemcpy(dPolygons, &dObjData[i].polygons, sizeof(Glpa::GPU_OBJECT3D_DATA*), cudaMemcpyDeviceToHost);
        err = cudaFree(dPolygons);
    }
    err = cudaFree(dObjData);
    objMtDataMalloced = false;
}

void Glpa::Render3d::dMallocObjInfo(std::unordered_map<std::string, Glpa::SceneObject *> &objs)
{
    cudaError_t err;
    if (objInfoMalloced)
    {
        dReleaseObjInfo();
    };

    std::vector<Glpa::GPU_OBJECT3D_INFO> hObjInfo;
    for (auto& pair : objIdMap)
    {
        if (Glpa::StationaryObject* obj = dynamic_cast<Glpa::StationaryObject*>(objs[pair.first]))
        {
            hObjInfo.push_back(obj->getInfo());
        }
        else
        {
            Glpa::outputErrorLog(__FILE__, __LINE__, {"Object is not a StationaryObject."});
        }
    }

    err = cudaMalloc(&dObjInfo, objIdMap.size() * sizeof(Glpa::GPU_OBJECT3D_INFO));
    err = cudaMemcpy(dObjInfo, hObjInfo.data(), objIdMap.size() * sizeof(Glpa::GPU_OBJECT3D_INFO), cudaMemcpyHostToDevice);

    objInfoMalloced = true;
}

void Glpa::Render3d::dReleaseObjInfo()
{
    cudaError_t err;
    if (!objInfoMalloced) return;

    err = cudaFree(dObjInfo);
    objInfoMalloced = false;
}

__global__ void GpuPrepareObj
(
    Glpa::GPU_OBJECT3D_DATA* objData,
    Glpa::GPU_OBJECT3D_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    int objAmount  
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Glpa::GPU_VECTOR_MG vecMg;

    if (i < objAmount)
    {
        // Get the object's existence range in the camera coordinate system
        Glpa::GPU_RANGE_RECT objRangeRect;
        for (int vI = 0; vI < 8; vI++)
        {
            objRangeRect.addRangeV(camData->mtTransRot.productLeft3x1(objData[i].range.wv[vI]));
        }

        // By looking two-dimensionally, 
        // it is possible to determine whether an object is even partially within the camera's viewing angle.
        Glpa::GPU_VEC_2D cullingVecs[4] = {
            {objRangeRect.origin.x, objRangeRect.opposite.z},
            {objRangeRect.opposite.x, objRangeRect.opposite.z},
            {objRangeRect.origin.y, objRangeRect.opposite.z},
            {objRangeRect.opposite.y, objRangeRect.opposite.z}
        };

        Glpa::GPU_VEC_2D axisVec(0, -1);

        float vecsCos[4] = {
            vecMg.cos(cullingVecs[0], axisVec),
            vecMg.cos(cullingVecs[1], axisVec),
            vecMg.cos(cullingVecs[2], axisVec),
            vecMg.cos(cullingVecs[3], axisVec)
        };

        GPU_BOOL isObjZIn = GPU_CO
        (
            objRangeRect.origin.z >= -camData->farZ && objRangeRect.opposite.z <= -camData->nearZ, 
            TRUE, FALSE
        );

        GPU_BOOL isObjXzIn = GPU_CO
        (
            (objRangeRect.origin.x >= 0 && vecsCos[0] >= camData->fovXzCos) || 
            (objRangeRect.opposite.x <= 0 && vecsCos[1] >= camData->fovXzCos) ||
            (objRangeRect.origin.x <= 0 && objRangeRect.opposite.x >= 0),
            TRUE, FALSE
        );

        GPU_BOOL isObjYzIn = GPU_CO
        (
            (objRangeRect.origin.y >= 0 && vecsCos[2] >= camData->fovYzCos) || 
            (objRangeRect.opposite.y <= 0 && vecsCos[3] >= camData->fovYzCos) ||
            (objRangeRect.origin.y <= 0 && objRangeRect.opposite.y >= 0),
            TRUE, FALSE
        );

        objInfo[i].isInVV = GPU_CO
        (
            isObjZIn == TRUE && isObjXzIn == TRUE && isObjYzIn == TRUE, 
            TRUE, FALSE
        );

    } // if (i < objAmount)
}

void Glpa::Render3d::prepareObjs()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = objIdMap.size();
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    GpuPrepareObj<<<dimGrid, dimBlock>>>(dObjData, dObjInfo, dCamData, dataSize);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != 0) Glpa::runTimeError(__FILE__, __LINE__, {"Processing with Cuda failed."});
}

__global__ void GpuSetVs
(
    Glpa::GPU_OBJECT3D_DATA* objData,
    Glpa::GPU_OBJECT3D_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    int objAmount  
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Glpa::GPU_VECTOR_MG vecMg;

    if (i < objAmount)
    {
        
    }


}

void Glpa::Render3d::setVs()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = objIdMap.size();
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    GpuSetVs<<<dimGrid, dimBlock>>>(dObjData, dObjInfo, dCamData, dataSize);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != 0) Glpa::runTimeError(__FILE__, __LINE__, {"Processing with Cuda failed."});
}

void Glpa::Render3d::rasterize()
{
}

void Glpa::Render3d::run(
    std::unordered_map<std::string, Glpa::SceneObject *> &objs, std::unordered_map<std::string, Glpa::Material *> &mts,
    Glpa::Camera &cam, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    dMallocCam(cam);
    if (!objMtDataMalloced) dMallocObjsMtData(objs, mts);
    dMallocObjInfo(objs);

    prepareObjs();
    setVs();
    rasterize();
}

void Glpa::Render3d::dRelease()
{
    dReleaseCam();
    dReleaseObjsMtData();
    dReleaseObjInfo();
}
