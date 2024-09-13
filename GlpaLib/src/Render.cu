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

void Glpa::Render3d::dMalloc
(
    Glpa::Camera &cam, 
    std::unordered_map<std::string,
    Glpa::SceneObject *> &objs, std::unordered_map<std::string, Glpa::Material *> &mts
){
    // Camera data
    if (camFactory.malloced) camFactory.dFree(dCamData);;
    camFactory.dMalloc(dCamData, cam.getData());

    // Material data
    if (!mtFactory.malloced) mtFactory.dMalloc(dMts, mts);;

    // Object data
    if (!stObjFactory.dataMalloced) stObjFactory.dMalloc(dStObjData, dObjPolys, objs, mtFactory.idMap);

    // Object info
    if (stObjFactory.infoMalloced) stObjFactory.dFree(dStObjInfo);
    stObjFactory.dMalloc(dStObjInfo, objs);

    // Result data
    if (resultFactory.malloced) resultFactory.dFree(dResult);
    resultFactory.dMalloc(dResult, stObjFactory.idMap.size());
}

__global__ void GpuPrepareObj
(
    Glpa::GPU_ST_OBJECT_DATA* objData,
    Glpa::GPU_ST_OBJECT_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    Glpa::GPU_RENDER_RESULT* result,
    int objAmount
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    Glpa::GPU_VECTOR_MG vecMgr;

    if (i < objAmount)
    {
        // Get the object's existence range in the camera coordinate system
        Glpa::GPU_RANGE_RECT objRangeRect;
        for (int vI = 0; vI < 8; vI++)
        {
            objRangeRect.addRangeV(camData->mtTransRot.productLeft3x1(objData[i].range.wv[vI]));
        }

        objInfo[i].isInVV = camData->isInside(objRangeRect);

        GPU_IF(objInfo[i].isInVV == TRUE, branch2)
        {
            atomicAdd(&result->objSum, 1);
            atomicAdd(&result->polySum, objData[i].polyAmount);

            result->dPolyAmounts[i] = objData[i].polyAmount;
        }
        
    } // if (i < objAmount)
}


void Glpa::Render3d::prepareObjs()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = stObjFactory.idMap.size();
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    GpuPrepareObj<<<dimGrid, dimBlock>>>(dStObjData, dStObjInfo, dCamData, dResult, dataSize);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != 0) Glpa::runTimeError(__FILE__, __LINE__, {"Processing with Cuda failed."});

    resultFactory.deviceToHost(dResult);
}

__device__ void GpuSetI(int nI, int* polyAmounts, int objSum, int& objI, int& polyI)
{
    int polyAmountSum = polyAmounts[0];
    for (int i = 1; i <= objSum; i++)
    {
        if (nI + 1 <= polyAmountSum)
        {
            objI = i - 1;
            polyI = nI - (polyAmountSum - polyAmounts[i - 1]);
            return;
        }
        else
        {
            polyAmountSum += polyAmounts[i];
        }
    }

    objI = GPU_IS_EMPTY;
    polyI = GPU_IS_EMPTY;
    return;
}

__device__ GPU_BOOL GpuGetFaceLineInxtn(Glpa::GPU_FACE_3D& face, Glpa::GPU_LINE_3D& line, Glpa::GPU_VEC_3D& inxtn)
{
    Glpa::GPU_VECTOR_MG vecMgr;

    /* 
    one vertex of the surface be p, 
    the normal vector of the surface be n,
    the starting point of the line segment be a, 
    the end point of the line segment be b.
    */
    Glpa::GPU_VEC_3D pa = vecMgr.getVec(face.v, line.start);
    Glpa::GPU_VEC_3D pb = vecMgr.getVec(face.v, line.end);

    // The dot product of the normal vector of the surface and the line segment
    float dotPaN = vecMgr.dot(face.n, pa);
    float dotPbN = vecMgr.dot(face.n, pb);

    // Determine whether they intersect.
    GPU_BOOL isIntersect = GPU_CO(dotPaN * dotPbN <= 0, TRUE, FALSE);

    GPU_IF(isIntersect == FALSE, br1) return FALSE;

    // Find the absolute value of the inner product.
    float absDotPaN = abs(dotPaN);
    float absDotPbN = abs(dotPbN);

    Glpa::GPU_VEC_3D intersectV = line.start + line.vec * (absDotPaN / (absDotPaN + absDotPbN));

    GPU_BOOL isInside = face.isInside(intersectV);
    GPU_IF(isInside == TRUE, br2)
    {
        inxtn = intersectV;
        return TRUE;
    }

    return FALSE;

}

__global__ void GpuSetVs
(
    Glpa::GPU_ST_OBJECT_DATA* objData,
    Glpa::GPU_POLYGON* objPolys,
    Glpa::GPU_ST_OBJECT_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    Glpa::GPU_RENDER_RESULT* result
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Glpa::GPU_VECTOR_MG vecMgr;

    if (i < result->polySum)
    {
        // Get the index of each object and polygon from the current i
        int objI, polyI;
        GpuSetI(i, result->dPolyAmounts, result->objSum, objI, polyI);

        // Execute if current i is not out of range.
        GPU_IF(objI != GPU_IS_EMPTY, br2)
        {
            result->facingObjI[i] = GPU_IS_EMPTY;
            result->facingPolyI[i] = GPU_IS_EMPTY;

            result->inxtnObjId[i] = GPU_IS_EMPTY;
            result->inxtnPolyId[i] = GPU_IS_EMPTY;
            result->inxtnAmountsPoly[i] = GPU_IS_EMPTY;
            result->inxtnAmountsVv[i] = GPU_IS_EMPTY;

            // Check if the polygon is facing the camera
            GPU_BOOL isPolyFacing = objPolys[i].isFacing(camData->mtTransRot, camData->mtRot);

            GPU_IF(isPolyFacing == TRUE, br3)
            {
                // Add the result to the result data
                atomicAdd(&result->facingPolySum, isPolyFacing);
                result->facingObjI[i] = objI;
                result->facingPolyI[i] = polyI;

                Glpa::GPU_POLYGON ctPoly(objPolys[i], camData->mtTransRot, camData->mtRot);

                GPU_BOOL isCtVIn[3];
                GPU_BOOL isPolyIn = FALSE;
                GPU_BOOL isPolyRangeIn = FALSE;
                int inVSum = 0;

                // Determine whether the polygon's vertices are within the view volume.
                for (int j = 0; j < 3; j++)
                {
                    isCtVIn[j] = camData->isInside(ctPoly.wv[j]);
                    isPolyIn = GPU_CO(isPolyIn + isCtVIn[j] >= 1, TRUE, FALSE);
                    inVSum += isCtVIn[j];
                }

                // Add the result to the result data
                atomicAdd(&result->insidePolySum, isPolyIn);
                GPU_IF(inVSum != 3 && isPolyIn == TRUE, br4)
                {
                    atomicAdd(&result->needClipPolySum, 1);
                }

                GPU_IF(isPolyIn == FALSE, br4)
                {
                    Glpa::GPU_RANGE_RECT polyRangeRect;
                    for (int j = 0; j < 3; j++)
                    {
                        polyRangeRect.addRangeV(ctPoly.wv[j]);
                    }
                    polyRangeRect.setWvs();

                    isPolyRangeIn = camData->isInside(polyRangeRect);
                }

                // Add the result to the result data
                atomicAdd(&result->insidePolyRangeSum, isPolyRangeIn);

                GPU_IF((inVSum != 3 && isPolyIn == TRUE) || isPolyRangeIn == TRUE, br4)
                {
                    Glpa::GPU_FACE_3D polyFace(ctPoly.wv[0], ctPoly.n);
                    polyFace.setTriangle(ctPoly.wv[0], ctPoly.wv[1], ctPoly.wv[2]);

                    Glpa::GPU_LINE_3D polyLine[3] = 
                    {
                        Glpa::GPU_LINE_3D(ctPoly.wv[0], ctPoly.wv[1]),
                        Glpa::GPU_LINE_3D(ctPoly.wv[1], ctPoly.wv[2]),
                        Glpa::GPU_LINE_3D(ctPoly.wv[2], ctPoly.wv[0])
                    };

                    int inxtnAmount = 0;

                    // Obtain the intersection of the polygon surface and the view volume line.
                    GPU_BOOL isExistAtPolyFace[GPU_VV_LINE_AMOUNT];
                    Glpa::GPU_VEC_3D polyFaceInxtn[GPU_VV_LINE_AMOUNT];
                    for (int j = 0; j < GPU_VV_LINE_AMOUNT; j++)
                    {
                        isExistAtPolyFace[j] = GpuGetFaceLineInxtn(polyFace, camData->vv.line[j], polyFaceInxtn[j]);
                        inxtnAmount += isExistAtPolyFace[j];
                    }

                    // Add the result to the result data
                    GPU_IF(inxtnAmount != 0, br5)
                    {
                        result->inxtnObjId[i] = objI;
                        result->inxtnPolyId[i] = polyI;
                        result->inxtnAmountsPoly[i] = inxtnAmount;
                    }


                    // Obtain the intersection of the view volume surface and the polygon line.
                    GPU_BOOL isExistAtVVFace[GPU_VV_FACE_AMOUNT][GPU_POLY_LINE_AMOUNT];
                    Glpa::GPU_VEC_3D vvFaceInxtn[GPU_VV_FACE_AMOUNT][GPU_POLY_LINE_AMOUNT];
                    for (int j = 0; j < GPU_VV_FACE_AMOUNT; j++)
                    {
                        for (int k = 0; k < GPU_POLY_LINE_AMOUNT; k++)
                        {
                            isExistAtVVFace[j][k] = GpuGetFaceLineInxtn(camData->vv.face[j], polyLine[k], vvFaceInxtn[j][k]);
                            inxtnAmount += isExistAtVVFace[j][k];
                        }
                    }

                    // Add the result to the result data
                    GPU_IF(inxtnAmount != 0, br5)
                    {
                        result->inxtnObjId[i] = objI;
                        result->inxtnPolyId[i] = polyI;
                        result->inxtnAmountsVv[i] = inxtnAmount;
                    }
                }
                
            }
        }
    }


}

void Glpa::Render3d::setVs()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = resultFactory.hResult.polySum;
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    GpuSetVs<<<dimGrid, dimBlock>>>(dStObjData, dObjPolys,dStObjInfo, dCamData, dResult);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != 0) Glpa::runTimeError(__FILE__, __LINE__, {"Processing with Cuda failed."});

    resultFactory.deviceToHost(dResult);
}

void Glpa::Render3d::rasterize()
{
}

void Glpa::Render3d::run(
    std::unordered_map<std::string, Glpa::SceneObject *> &objs, std::unordered_map<std::string, Glpa::Material *> &mts,
    Glpa::Camera &cam, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi)
{
    dMalloc(cam, objs, mts);

    prepareObjs();
    setVs();
    rasterize();
}

void Glpa::Render3d::dRelease()
{
    camFactory.dFree(dCamData);
    mtFactory.dFree(dMts);
    stObjFactory.dFree(dStObjData, dObjPolys);
    stObjFactory.dFree(dStObjInfo);
    resultFactory.dFree(dResult);
}

void Glpa::RENDER_RESULT_FACTORY::dFree(Glpa::GPU_RENDER_RESULT*& dResult)
{
    if (!malloced) return;

    delete[] dResult->hPolyAmounts;
    dResult->hPolyAmounts = nullptr;

    int* dPolyAmounts;
    cudaMemcpy(&dPolyAmounts, &dResult->dPolyAmounts, sizeof(int*), cudaMemcpyDeviceToHost);
    cudaFree(dPolyAmounts);

    free(dResult);
    dResult = nullptr;
    malloced = false;
}

void Glpa::RENDER_RESULT_FACTORY::dMalloc(Glpa::GPU_RENDER_RESULT*& dResult, int srcObjSum)
{
    if (malloced) dFree(dResult);

    hResult.srcObjSum = srcObjSum;
    hResult.objSum = 0;
    hResult.polySum = 0;

    hResult.facingPolySum = 0;
    hResult.insidePolySum = 0;
    hResult.insidePolyRangeSum = 0;

    hResult.needClipPolySum = 0;

    hResult.polyFaceInxtnSum = 0;
    hResult.vvFaceInxtnSum = 0;

    hResult.hPolyAmounts = new int[srcObjSum];
    cudaMalloc(&hResult.dPolyAmounts, srcObjSum * sizeof(int));

    cudaMalloc(&dResult, sizeof(Glpa::GPU_RENDER_RESULT));
    cudaMemcpy(dResult, &hResult, sizeof(Glpa::GPU_RENDER_RESULT), cudaMemcpyHostToDevice);

    cudaFree(hResult.dPolyAmounts);

    malloced = true;
}

void Glpa::RENDER_RESULT_FACTORY::deviceToHost(Glpa::GPU_RENDER_RESULT*& dResult)
{
    if (!malloced) Glpa::runTimeError(__FILE__, __LINE__, {"There is no memory on the result device side."});

    cudaMemcpy(&hResult, dResult, sizeof(Glpa::GPU_RENDER_RESULT), cudaMemcpyDeviceToHost);

    int* dOtherPolyAmounts;
    cudaMalloc(&dOtherPolyAmounts, hResult.srcObjSum * sizeof(int));
    cudaMemcpy(dOtherPolyAmounts, hResult.dPolyAmounts, hResult.srcObjSum * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMemcpy(hResult.hPolyAmounts, dOtherPolyAmounts, hResult.srcObjSum * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dOtherPolyAmounts);
}
