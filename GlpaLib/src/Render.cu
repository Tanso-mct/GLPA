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
        cudaDeviceSynchronize();
        checkCudaErr(__FILE__, __LINE__, __FUNCSIG__);

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
    std::unordered_map<std::string,Glpa::SceneObject *> &objs, std::unordered_map<std::string, Glpa::Material *> &mts,
    int& bufWidth, int& bufHeight, int& bufDpi
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
    if (resultFactory.malloced) resultFactory.dFree(dResult, dPolyAmounts);
    resultFactory.dMalloc(dResult, dPolyAmounts, stObjFactory.idMap.size(), bufWidth, bufHeight, bufDpi);

    if (dBuf == nullptr) cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
    cudaMemset(dBuf, 0, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
}

__global__ void GpuPrepareObj
(
    Glpa::GPU_ST_OBJECT_DATA* objData,
    Glpa::GPU_ST_OBJECT_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    Glpa::GPU_RENDER_RESULT* result,
    int* polyAmounts,
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

            polyAmounts[i] = objData[i].polyAmount;
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

    GpuPrepareObj<<<dimGrid, dimBlock>>>(dStObjData, dStObjInfo, dCamData, dResult, dPolyAmounts, dataSize);
    cudaDeviceSynchronize();
    checkCudaErr(__FILE__, __LINE__, __FUNCSIG__);

    resultFactory.deviceToHost(dResult, dPolyAmounts);
}

__device__ void GpuSetI(int nI, int* polyAmounts, int objSum, int& objId, int& polyId)
{
    int polyAmountSum = polyAmounts[0];
    for (int i = 1; i <= objSum; i++)
    {
        GPU_IF(nI + 1 <= polyAmountSum, br1)
        {
            objId = i - 1;
            polyId = nI - (polyAmountSum - polyAmounts[i - 1]);
            return;
        }
        GPU_IF(nI + 1 > polyAmountSum, br1)
        {
            polyAmountSum += polyAmounts[i];
        }
    }

    objId = GPU_IS_EMPTY;
    polyId = GPU_IS_EMPTY;
    return;
}

__device__ GPU_BOOL GpuGetFaceLineInxtn(Glpa::GPU_FACE_3D& face, Glpa::GPU_LINE_3D& line, Glpa::GPU_ARRAY& polyVs)
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
        polyVs.push(intersectV);
        return TRUE;
    }

    return FALSE;

}

__device__ void GpuSetMaxMinY(int& maxY, int& minY, int val)
{
    maxY = GPU_CO(val > maxY, val, maxY);
    minY = GPU_CO(val < minY, val, minY);
}

__device__ void GpuSetMaxMinCoords(Glpa::GPU_VEC_2D* maxCoords, Glpa::GPU_VEC_2D* minCoords, Glpa::GPU_VEC_3D* coord, int i)
{
    GPU_BOOL isMaxX = GPU_CO(maxCoords[i].x < coord->x, TRUE, FALSE);
    maxCoords[i].x = GPU_CO(isMaxX == TRUE, coord->x, maxCoords[i].x);
    maxCoords[i].y = GPU_CO(isMaxX == TRUE, coord->z, maxCoords[i].y);

    GPU_BOOL isMinX = GPU_CO(minCoords[i].x > coord->x, TRUE, FALSE);
    minCoords[i].x = GPU_CO(isMinX == TRUE, coord->x, minCoords[i].x);
    minCoords[i].y = GPU_CO(isMinX == TRUE, coord->z, minCoords[i].y);
}

__global__ void GpuPrepareLines
(
    Glpa::GPU_ST_OBJECT_DATA* objData,
    Glpa::GPU_POLYGON** objPolys,
    Glpa::GPU_ST_OBJECT_INFO* objInfo,
    Glpa::GPU_CAMERA* camData,
    Glpa::GPU_POLY_LINE** mPolyLines,
    Glpa::GPU_MPOLYGON_INFO* mPolyInfo,
    Glpa::GPU_RENDER_RESULT* result,
    int* polyAmounts
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Glpa::GPU_VECTOR_MG vecMgr;

    if (i < result->polySum)
    {
        // Get the index of each object and polygon from the current i
        int objId, polyId;
        GpuSetI(i, polyAmounts, result->objSum, objId, polyId);

        mPolyInfo[i].height = 0;

        // Execute if current i is not out of range.
        GPU_IF(objId != GPU_IS_EMPTY, br2)
        {
            // Add the result to the result data
            // result->facingObjI[i] = GPU_IS_EMPTY;
            // result->facingPolyI[i] = GPU_IS_EMPTY;
            // result->inxtnObjId[i] = GPU_IS_EMPTY;
            // result->inxtnPolyId[i] = GPU_IS_EMPTY;
            // result->inxtnAmountsPoly[i] = GPU_IS_EMPTY;
            // result->inxtnAmountsVv[i] = GPU_IS_EMPTY;
            // for (int j = 0; j < 12; j++)
            // {
            //     for (int k = 0; k < 7; k++)
            //     {
            //         result->mPolyCubeVs[i][k][Glpa::X] = GPU_IS_EMPTY;
            //         result->mPolyCubeVs[i][k][Glpa::Y] = GPU_IS_EMPTY;
            //         result->mPolyCubeVs[i][k][Glpa::Z] = GPU_IS_EMPTY;
            //     }
            // }
            // for (int j = 0; j < 200; j++)
            // {
            //     for (int k = 0; k < 7; k++)
            //     {
            //         result->mPolyPlaneVs[i-12][k][Glpa::X] = GPU_IS_EMPTY;
            //         result->mPolyPlaneVs[i-12][k][Glpa::Y] = GPU_IS_EMPTY;
            //         result->mPolyPlaneVs[i-12][k][Glpa::Z] = GPU_IS_EMPTY;
            //     }
            // }

            // Check if the polygon is facing the camera
            GPU_BOOL isPolyFacing = objPolys[objId][polyId].isFacing(camData->mtTransRot, camData->mtRot);

            GPU_IF(isPolyFacing == TRUE, br3)
            {
                // Add the result to the result data
                atomicAdd(&result->facingPolySum, isPolyFacing);
                result->facingObjI[i] = objId;
                result->facingPolyI[i] = polyId;

                // Get the polygon's coordinates in the camera coordinate system
                objPolys[objId][polyId].convert(camData->mtTransRot, camData->mtRot);

                GPU_BOOL isCtVIn[3];
                GPU_BOOL isPolyIn = FALSE;
                GPU_BOOL isPolyRangeIn = FALSE;
                int inVSum = 0;

                // Determine whether the polygon's vertices are within the view volume.
                for (int j = 0; j < 3; j++)
                {
                    isCtVIn[j] = camData->isInside(objPolys[objId][polyId].ctWv[j]);
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
                        polyRangeRect.addRangeV(objPolys[objId][polyId].ctWv[j]);
                    }
                    polyRangeRect.setWvs();

                    isPolyRangeIn = camData->isInside(polyRangeRect);
                }

                // // Add the result to the result data
                // atomicAdd(&result->insidePolyRangeSum, isPolyRangeIn);

                // Store the coordinates of the intersection
                Glpa::GPU_ARRAY mPolyVs;

                for (int j = 0; j < 3; j++)
                {
                    GPU_IF(isCtVIn[j] == TRUE, br6)
                    {
                        mPolyVs.push(objPolys[objId][polyId].ctWv[j]);
                        result->onErr = GPU_CO(mPolyVs.size >= 7, TRUE, FALSE);
                    }
                }

                GPU_IF((inVSum != 3 && isPolyIn == TRUE) || isPolyRangeIn == TRUE, br4)
                {
                    Glpa::GPU_FACE_3D polyFace(objPolys[objId][polyId].ctWv[0], objPolys[objId][polyId].ctN);
                    polyFace.setTriangle(objPolys[objId][polyId].ctWv[0], objPolys[objId][polyId].ctWv[1], objPolys[objId][polyId].ctWv[2]);

                    Glpa::GPU_LINE_3D polyLine[3] = 
                    {
                        Glpa::GPU_LINE_3D(objPolys[objId][polyId].ctWv[0], objPolys[objId][polyId].ctWv[1]),
                        Glpa::GPU_LINE_3D(objPolys[objId][polyId].ctWv[1], objPolys[objId][polyId].ctWv[2]),
                        Glpa::GPU_LINE_3D(objPolys[objId][polyId].ctWv[2], objPolys[objId][polyId].ctWv[0])
                    };

                    int inxtnAmount = 0;

                    // Obtain the intersection of the polygon surface and the view volume line.
                    for (int j = 0; j < GPU_VV_LINE_AMOUNT; j++)
                    {
                        GpuGetFaceLineInxtn(polyFace, camData->vv.line[j], mPolyVs);
                    }

                    // Add the result to the result data
                    // GPU_IF(inxtnAmount != 0, br5)
                    // {
                    //     result->inxtnObjId[i] = objId;
                    //     result->inxtnPolyId[i] = polyId;
                    //     result->inxtnAmountsPoly[i] = inxtnAmount;
                    // }

                    // Obtain the intersection of the view volume surface and the polygon line.
                    for (int j = 0; j < GPU_VV_FACE_AMOUNT; j++)
                    {
                        for (int k = 0; k < GPU_POLY_LINE_AMOUNT; k++)
                        {
                            GpuGetFaceLineInxtn(camData->vv.face[j], polyLine[k], mPolyVs);
                        }
                    }

                    // Add the result to the result data
                    // GPU_IF(inxtnAmount != 0, br5)
                    // {
                    //     result->inxtnObjId[i] = objId;
                    //     result->inxtnPolyId[i] = polyId;
                    //     result->inxtnAmountsVv[i] = inxtnAmount;
                    // }

                }

                // Add the result to the result data
                // GPU_IF(mPolyVs.size != 0, br4)
                // {
                //     GPU_IF(i < 12, br5)
                //     {
                //         for (int j = 0; j < mPolyVs.size; j++)
                //         {
                //             Glpa::GPU_VEC_3D* mPolyV = mPolyVs.get(j);
                //             result->mPolyCubeVs[i][j][Glpa::X] = mPolyV->x;
                //             result->mPolyCubeVs[i][j][Glpa::Y] = mPolyV->y;
                //             result->mPolyCubeVs[i][j][Glpa::Z] = mPolyV->z;
                //         }
                //     }
                //     GPU_IF(i >= 12 && i < 212, br5)
                //     {
                //         for (int j = 0; j < mPolyVs.size; j++)
                //         {
                //             Glpa::GPU_VEC_3D* mPolyV = mPolyVs.get(j);
                //             result->mPolyPlaneVs[i-12][j][Glpa::X] = mPolyV->x;
                //             result->mPolyPlaneVs[i-12][j][Glpa::Y] = mPolyV->y;
                //             result->mPolyPlaneVs[i-12][j][Glpa::Z] = mPolyV->z;
                //         }
                //     }
                // }

                // // Arrange vertices in the order they form a line segment
                GPU_IF(mPolyVs.size >= 3, br4)
                {
                    Glpa::GPU_ARRAY mPolyScrVs;
                    for (int j = 0; j < mPolyVs.size; j++)
                    {
                        mPolyScrVs.push(camData->getScrPos(*mPolyVs.get(j)));
                    }

                    Glpa::GPU_VEC_2D baseVec = vecMgr.getVec
                    (
                        {mPolyScrVs.get(0)->x, mPolyScrVs.get(0)->y}, {mPolyScrVs.get(1)->x, mPolyScrVs.get(1)->y}
                    );

                    float compareCross[5];
                    for (int j = 0; j < mPolyVs.size - 2; j++)
                    {
                        compareCross[j] = vecMgr.cross
                        (
                            baseVec, 
                            vecMgr.getVec
                            (
                                {mPolyScrVs.get(0)->x, mPolyScrVs.get(0)->y}, 
                                {mPolyScrVs.get(j+2)->x, mPolyScrVs.get(j+2)->y}
                            )
                        );
                    }

                    // Sort the vertices based on the size of the cross product.
                    Glpa::GPU_LIST3 leftVs;
                    Glpa::GPU_LIST3 rightVs;

                    for (int j = 0; j < mPolyVs.size - 2; j++)
                    {
                        GPU_IF(compareCross[j] < 0, br5)
                        {
                            leftVs.push
                            ({
                                j+2,
                                vecMgr.dot
                                (
                                    baseVec, 
                                    vecMgr.getVec
                                    (
                                        {mPolyScrVs.get(0)->x, mPolyScrVs.get(0)->y}, 
                                        {mPolyScrVs.get(j+2)->x, mPolyScrVs.get(j+2)->y}
                                    )
                                )
                            });
                        }
                        GPU_IF(compareCross[j] >= 0, br5)
                        {
                            rightVs.push
                            ({
                                j+2, 
                                vecMgr.dot
                                (
                                    baseVec, 
                                    vecMgr.getVec
                                    (
                                        {mPolyScrVs.get(0)->x, mPolyScrVs.get(0)->y}, 
                                        {mPolyScrVs.get(j+2)->x, mPolyScrVs.get(j+2)->y}
                                    )
                                )
                            });
                        }
                    }

                    leftVs.aSortByVal2();
                    rightVs.dSortByVal2();

                    // Store vertices in order.
                    Glpa::GPU_ARRAY sortedMPolyVs;
                    int maxY = 0;
                    int minY = camData->scrSize.y;

                    sortedMPolyVs.push(*mPolyScrVs.get(0));
                    GpuSetMaxMinY(maxY, minY, mPolyScrVs.get(0)->y);

                    for (int j = 0; j < rightVs.size; j++)
                    {
                        sortedMPolyVs.push(*mPolyScrVs.get((int)rightVs.pair[j].val1));
                        GpuSetMaxMinY(maxY, minY, mPolyScrVs.get((int)rightVs.pair[j].val1)->y);
                    }

                    sortedMPolyVs.push(*mPolyScrVs.get(1));
                    GpuSetMaxMinY(maxY, minY, mPolyScrVs.get(1)->y);

                    for (int j = 0; j < leftVs.size; j++)
                    {
                        sortedMPolyVs.push(*mPolyScrVs.get((int)leftVs.pair[j].val1));
                        GpuSetMaxMinY(maxY, minY, mPolyScrVs.get((int)leftVs.pair[j].val1)->y);
                    }

                    mPolyInfo[i].minY = minY;

                    atomicMax(&result->maxLineAmount, sortedMPolyVs.size);
                    mPolyInfo[i].lineAmount = sortedMPolyVs.size;

                    atomicMax(&result->maxPolyHeight, maxY - minY + 1);
                    mPolyInfo[i].height = maxY - minY + 1;

                    mPolyLines[i] = (Glpa::GPU_POLY_LINE*)malloc(sizeof(Glpa::GPU_POLY_LINE) * (sortedMPolyVs.size));

                    for (int j = 0; j < sortedMPolyVs.size-1; j++)
                    {
                        mPolyLines[i][j].set
                        (
                            objId, polyId, j, *sortedMPolyVs.get(j), j+1, *sortedMPolyVs.get(j+1)
                        );
                    }

                    mPolyLines[i][sortedMPolyVs.size-1].set
                    (
                        objId, polyId, 
                        sortedMPolyVs.size-1, *sortedMPolyVs.get(sortedMPolyVs.size-1), 
                        0, *sortedMPolyVs.get(0)
                    );

                    // Add the result to the result data
                    GPU_IF(sortedMPolyVs.size != 0, br4)
                    {
                        GPU_IF(i < 12, br5)
                        {
                            for (int j = 0; j < sortedMPolyVs.size; j++)
                            {
                                Glpa::GPU_VEC_3D* mPolyV = &mPolyLines[i][j].start;
                                result->mPolyCubeVs[i][j][Glpa::X] = mPolyV->x;
                                result->mPolyCubeVs[i][j][Glpa::Y] = mPolyV->y;
                                result->mPolyCubeVs[i][j][Glpa::Z] = mPolyV->z;
                            }
                        }

                        GPU_IF(i >= 12 && i < 212, br5)
                        {
                            for (int j = 0; j < sortedMPolyVs.size; j++)
                            {
                                Glpa::GPU_VEC_3D* mPolyV = &mPolyLines[i][j].start;
                                result->mPolyPlaneVs[i-12][j][Glpa::X] = mPolyV->x;
                                result->mPolyPlaneVs[i-12][j][Glpa::Y] = mPolyV->y;
                                result->mPolyPlaneVs[i-12][j][Glpa::Z] = mPolyV->z;
                            }
                        }
                    }

                    GPU_IF(i == 12 + 86, br5)
                    {
                        int debugNum = 0;
                        debugNum = 100;
                    }

                    mPolyScrVs.clear();
                    sortedMPolyVs.clear();

                    // int ySize = maxY - minY + 1;
                    // Glpa::GPU_VEC_2D* maxCoords = new Glpa::GPU_VEC_2D[ySize];
                    // Glpa::GPU_VEC_2D* minCoords = new Glpa::GPU_VEC_2D[ySize];
                    // for (int j = 0; j < ySize; j++)
                    // {
                    //     maxCoords[j].set(-1, -1);
                    //     minCoords[j].set(camData->scrSize.x + 1, camData->scrSize.y + 1);
                    // }

                    // for (int j = 0; j < sortedMPolyVs.size; j++)
                    // {
                    //     GpuSetMaxMinCoords(maxCoords, minCoords, sortedMPolyVs.get(j), sortedMPolyVs.get(j)->y - minY);
                    // }

                    // for (int j = 0; j < sortedMPolyVs.size-1; j++)
                    // {
                    //     int direction = GPU_CO(sortedMPolyVs.get(j+1)->y - sortedMPolyVs.get(j)->y >= 0, 1, -1);
                    //     int startY = sortedMPolyVs.get(j)->y;
                    //     int height = abs(sortedMPolyVs.get(j+1)->y - sortedMPolyVs.get(j)->y + 1);

                    //     for (int nY = 1; nY < height; nY++)
                    //     {
                    //         int diffY = startY + nY * direction;
                    //         Glpa::GPU_VEC_3D rasterizedV = GpuLineIP_Y
                    //         (
                    //             sortedMPolyVs.get(j), sortedMPolyVs.get(j+1), diffY
                    //         );

                    //         GpuSetMaxMinCoords(maxCoords, minCoords, &rasterizedV, diffY - minY - 1);
                    //     }
                    // }

                    // int startPointI = sortedMPolyVs.size - 1;
                    // int direction = GPU_CO(sortedMPolyVs.get(startPointI)->y - sortedMPolyVs.get(0)->y >= 0, 1, -1);
                    // int startY = sortedMPolyVs.get(0)->y;
                    // int height = abs(sortedMPolyVs.get(startPointI)->y - sortedMPolyVs.get(0)->y + 1);

                    // for (int nY = 1; nY < height; nY++)
                    // {
                    //     int diffY = startY + nY * direction;
                    //     Glpa::GPU_VEC_3D rasterizedV = GpuLineIP_Y
                    //     (
                    //         sortedMPolyVs.get(0), sortedMPolyVs.get(startPointI), diffY
                    //     );

                    //     GpuSetMaxMinCoords(maxCoords, minCoords, &rasterizedV, diffY - minY - 1);
                    // }

                    // for (int by = minY; by < minY + ySize; by++)
                    // {
                    //     for (int bx = minCoords[by-minY].x; bx < maxCoords[by-minY].x; bx++)
                    //     {
                    //         Glpa::GPU_VEC_3D startPoint = {minCoords[by-minY].x, by, minCoords[by-minY].y};
                    //         Glpa::GPU_VEC_3D endPoint = {maxCoords[by-minY].x, by, maxCoords[by-minY].y};
                    //         Glpa::GPU_VEC_3D thisPoint = GpuLineIP_X
                    //         (
                    //             &startPoint, &endPoint, bx
                    //         );

                    //         int zBufI = bx + by * bufWidth * bufDpi;
                    //         atomicExch((unsigned int*)&buf[zBufI], 0xFF0000FF);
                    //         // atomicExch(&zBufAry[zBufI].isEmpt, TRUE);buf[zBufI] = 0xFF0000FF;

                    //         // atomicExch(&zBufAry[zBufI].isEmpt, TRUE);
                    //         // zBufAry[zBufI].set(objId, polyId, thisPoint.z, {bx, by});
                            
                    //     }
                    // }

                    // delete[] maxCoords;
                    // delete[] minCoords;

                }

                mPolyVs.clear();


            }
        }
    }


}

__device__ Glpa::GPU_VEC_3D GpuLineIP_X(Glpa::GPU_VEC_3D* start, Glpa::GPU_VEC_3D* end, int x)
{
    Glpa::GPU_VECTOR_MG vecMgr;

    float t = (x - start->x) / (end->x - start->x);
    int y = start->y + t * (end->y - start->y);
    float z = start->z + t * (end->z - start->z);

    return Glpa::GPU_VEC_3D((float)x, (float)y, z);
}

__device__ Glpa::GPU_VEC_3D GpuLineIP_Y(Glpa::GPU_VEC_3D* start, Glpa::GPU_VEC_3D* end, int y)
{
    Glpa::GPU_VECTOR_MG vecMgr;

    float t = (y - start->y) / (end->y - start->y);
    int x = start->x + t * (end->x - start->x);
    float z = start->z + t * (end->z - start->z);

    return Glpa::GPU_VEC_3D((float)x, (float)y, z);
}

__device__ void GpuScanLines
(
    int lineY, Glpa::GPU_POLY_LINE* polyLines, int lineAmount, 
    Glpa::GPU_VEC_3D& leftPoint, Glpa::GPU_VEC_3D& rightPoint, int bufWidth, int bufDpi
){
    int leftLineI, rightLineI;
    float leftX = bufWidth * bufDpi;
    float rightX = 0;

    for (int i = 0; i < lineAmount; i++)
    {
        GPU_IF
        (
            (lineY >= polyLines[i].start.y && lineY <= polyLines[i].end.y) ||
            (lineY >= polyLines[i].end.y && lineY <= polyLines[i].start.y), 
            br1
        ){
            float m = (polyLines[i].end.y - polyLines[i].start.y) / (polyLines[i].end.x - polyLines[i].start.x);
            float x = polyLines[i].start.x + (lineY - polyLines[i].start.y) / m;

            GPU_IF(x < leftX, br2)
            {
                leftX = x;
                leftLineI = i;
            }

            GPU_IF(x > rightX, br2)
            {
                rightX = x;
                rightLineI = i;
            }
        }
    }

    leftPoint = GpuLineIP_X(&polyLines[leftLineI].start, &polyLines[leftLineI].end, leftX);
    rightPoint = GpuLineIP_X(&polyLines[rightLineI].start, &polyLines[rightLineI].end, rightX);

    leftPoint = {(int)leftPoint.x, (int)leftPoint.y, leftPoint.z};
    rightPoint = {(int)rightPoint.x, (int)rightPoint.y, rightPoint.z};
}

// __device__ void GpuPointDraw(int x, int y, int bufWidth, int bufHeight, int bufDpi, LPDWORD buf)
// {
//     int zBufI = x + y * bufWidth * bufDpi;

//     GPU_IF(zBufI >= 0 && zBufI < bufWidth * bufHeight * bufDpi, br1)
//     {
//         atomicExch((unsigned int*)&buf[zBufI], 0xFFFF0000);
//     }
// }

__global__ void GpuRasterize
(
    Glpa::GPU_POLY_LINE** polyLines, Glpa::GPU_MPOLYGON_INFO* mPolyInfo,
    int bufWidth, int bufHeight, int bufDpi, LPDWORD buf,
    Glpa::GPU_RENDER_RESULT* result
){
    int tI = blockIdx.y * blockDim.y + threadIdx.y;
    int tJ = blockIdx.x * blockDim.x + threadIdx.x;

    if (tI < result->polySum)
    {
        if (tJ < mPolyInfo[tI].height)
        {
            Glpa::GPU_VEC_3D leftPoint, rightPoint;
            GpuScanLines(mPolyInfo[tI].minY + tJ, polyLines[tI], mPolyInfo[tI].lineAmount, leftPoint, rightPoint, bufWidth, bufDpi);

            for (int i = leftPoint.x; i < rightPoint.x; i++)
            {
                int zBufI = i + (mPolyInfo[tI].minY + tJ) * bufWidth * bufDpi;
                atomicExch((unsigned int*)&buf[zBufI], 0xFF0000FF);
            }
        }
    }

    // __syncthreads();

    // if (tI < result->polySum)
    // {
    //     GPU_IF(mPolyInfo[tI].height != 0, br2)
    //     {
    //         for (int i = 0; i < mPolyInfo[tI].lineAmount; i++)
    //         {
    //             // Start
    //             GpuPointDraw((int)polyLines[tI][i].start.x, (int)polyLines[tI][i].start.y, bufWidth, bufHeight, bufDpi, buf);
    //             for (int j = 1; j < 5; j++)
    //             {
    //                 GpuPointDraw((int)polyLines[tI][i].start.x + j, (int)polyLines[tI][i].start.y, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].start.x - j, (int)polyLines[tI][i].start.y, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].start.x, (int)polyLines[tI][i].start.y + j, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].start.x, (int)polyLines[tI][i].start.y - j, bufWidth, bufHeight, bufDpi, buf);
    //             }

    //             // End
    //             GpuPointDraw((int)polyLines[tI][i].end.x, (int)polyLines[tI][i].end.y, bufWidth, bufHeight, bufDpi, buf);
    //             for (int j = 1; j < 5; j++)
    //             {
    //                 GpuPointDraw((int)polyLines[tI][i].end.x + j, (int)polyLines[tI][i].end.y, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].end.x - j, (int)polyLines[tI][i].end.y, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].end.x, (int)polyLines[tI][i].end.y + j, bufWidth, bufHeight, bufDpi, buf);
    //                 GpuPointDraw((int)polyLines[tI][i].end.x, (int)polyLines[tI][i].end.y - j, bufWidth, bufHeight, bufDpi, buf);
    //             }
    //         }
    //     }
    // }
}

void Glpa::Render3d::prepareLines()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = resultFactory.hResult.polySum;
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    if(stObjFactory.polyLineMalloced) stObjFactory.dFree(dPolyLines, dMPolyInfo);
    stObjFactory.dMalloc(dPolyLines, dMPolyInfo, resultFactory.hResult.polySum);

    GpuPrepareLines<<<dimGrid, dimBlock>>>
    (
        dStObjData, dObjPolys, dStObjInfo, dCamData, 
        dPolyLines, dMPolyInfo, dResult, dPolyAmounts
    );
    cudaDeviceSynchronize();
    checkCudaErr(__FILE__, __LINE__, __FUNCSIG__);

    resultFactory.deviceToHost(dResult, dPolyAmounts);
}

void Glpa::Render3d::rasterize(int& bufWidth, int& bufHeight, int& bufDpi, LPDWORD buf)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSizeY = resultFactory.hResult.polySum;
    int dataSizeX = resultFactory.hResult.maxPolyHeight;

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

    cudaMemset(dBuf, 0, bufWidth * bufHeight * bufDpi * sizeof(DWORD));

    GpuRasterize<<<dimGrid, dimBlock>>>
    (
        dPolyLines, dMPolyInfo, bufWidth, bufHeight, bufDpi, dBuf, dResult
    );
    cudaDeviceSynchronize();
    checkCudaErr(__FILE__, __LINE__, __FUNCSIG__);

    resultFactory.deviceToHost(dResult, dPolyAmounts);

    cudaMemcpy(buf, dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyDeviceToHost);
}

void Glpa::Render3d::run(
    std::unordered_map<std::string, Glpa::SceneObject *> &objs, std::unordered_map<std::string, Glpa::Material *> &mts,
    Glpa::Camera &cam, LPDWORD buf, int& bufWidth, int& bufHeight, int& bufDpi)
{
    dMalloc(cam, objs, mts, bufWidth, bufHeight, bufDpi);

    prepareObjs();
    prepareLines();
    rasterize(bufWidth, bufHeight, bufDpi, buf);
}

void Glpa::Render3d::dRelease()
{
    camFactory.dFree(dCamData);
    mtFactory.dFree(dMts);
    stObjFactory.dFree(dStObjData, dObjPolys);
    stObjFactory.dFree(dStObjInfo);
    stObjFactory.dFree(dPolyLines, dMPolyInfo);
    resultFactory.dFree(dResult, dPolyAmounts);
    cudaFree(dBuf);
}

void Glpa::RENDER_RESULT_FACTORY::dFree(Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts)
{
    if (!malloced) return;

    delete[] hPolyAmounts;
    cudaFree(dPolyAmounts);
    dPolyAmounts = nullptr;

    cudaFree(dResult);
    dResult = nullptr;

    malloced = false;
}

void Glpa::RENDER_RESULT_FACTORY::dMalloc
(
    Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts, int srcObjSum, 
    int bufWidth, int bufHeight, int bufDpi
){
    if (malloced) return;

    hResult.srcObjSum = srcObjSum;
    hResult.objSum = 0;
    hResult.polySum = 0;

    hResult.facingPolySum = 0;
    hResult.insidePolySum = 0;
    hResult.insidePolyRangeSum = 0;

    hResult.needClipPolySum = 0;

    hResult.polyFaceInxtnSum = 0;
    hResult.vvFaceInxtnSum = 0;

    hResult.maxLineAmount = 0;
    hResult.maxPolyHeight = 0;

    hResult.onErr = FALSE;
    hResult.debugNum = 0;

    hResult.rasterizePolySum = 0;

    cudaMalloc(&dResult, sizeof(Glpa::GPU_RENDER_RESULT));
    cudaMemcpy(dResult, &hResult, sizeof(Glpa::GPU_RENDER_RESULT), cudaMemcpyHostToDevice);

    hPolyAmounts = new int[srcObjSum];
    cudaMalloc(&dPolyAmounts, srcObjSum * sizeof(int));
    cudaMemset(dPolyAmounts, 0, srcObjSum * sizeof(int));

    malloced = true;
}

void Glpa::RENDER_RESULT_FACTORY::deviceToHost(Glpa::GPU_RENDER_RESULT*& dResult, int*& dPolyAmounts)
{
    if (!malloced) Glpa::runTimeError(__FILE__, __LINE__, {"There is no memory on the result device side."});

    cudaMemcpy(&hResult, dResult, sizeof(Glpa::GPU_RENDER_RESULT), cudaMemcpyDeviceToHost);
    cudaMemcpy(hPolyAmounts, dPolyAmounts, hResult.srcObjSum * sizeof(int), cudaMemcpyDeviceToHost);
}
