#include "Render.cuh"
#include "GlpaLog.h"

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
            OutputDebugStringA("GlpaLib ERROR Render.cu - Processing with Cuda failed.\n");
            throw std::runtime_error("Processing with Cuda failed.");
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

void Glpa::Render3d::run(
    std::unordered_map<std::string, Glpa::SceneObject*> objs, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi
){

}

__global__ void Glpa::Gpu2dDraw
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
