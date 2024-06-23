#include "Render.cuh"

Glpa::Render2d::Render2d()
{
}

Glpa::Render2d::~Render2d()
{
}

void Glpa::Render2d::setBackground(std::string color, DWORD& bg)
{
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
    delete hBuf;
    cudaFree(dBuf);

    hBuf = new DWORD[bufWidth * bufHeight * bufDpi];
    std::fill(hBuf, hBuf + bufWidth * bufHeight * bufDpi, backgroundColor);

    cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
}

void Glpa::Render2d::dMalloc
(
    std::unordered_map<std::string, Glpa::SceneObject*>& objs,
    std::map<int, std::vector<std::string>>& drawOrder,
    int bufWidth, int bufHeight, int bufDpi, std::string bgColor
){
    if (malloced) return;
    
    hImgPosX.clear();
    hImgPosY.clear();
    hImgWidth.clear();
    hImgHeight.clear();
    hImgData.clear();

    for (auto& pair : drawOrder)
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
                    hImgWidth.push_back(img->getWidth());
                    hImgHeight.push_back(img->getHeight());

                    imgNames.push_back(img->getName());

                    maxImgWidth = (maxImgWidth < img->getWidth()) ? img->getWidth() : maxImgWidth;
                    maxImgHeight = (maxImgHeight < img->getHeight()) ? img->getHeight() : maxImgHeight;

                    DWORD* dData;
                    size_t dataSize = img->getWidth() * img->getHeight() * sizeof(DWORD);
                    cudaMalloc(&dData, dataSize);
                    cudaMemcpy(dData, img->getData(), dataSize, cudaMemcpyHostToDevice);
                    hImgData.push_back(dData);
                }
            }
        }
    }

    imgAmount = hImgData.size();

    setBackground(bgColor, backgroundColor);

    hBuf = new DWORD[bufWidth * bufHeight * bufDpi];
    std::fill(hBuf, hBuf + bufWidth * bufHeight * bufDpi, backgroundColor);

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


    for (int i = 0; i < hImgData.size(); i++)
    {
        DWORD* ptDeviceData;
    
        cudaMemcpy(&ptDeviceData, &dImgData[i], sizeof(DWORD*), cudaMemcpyDeviceToHost);

        cudaFree(ptDeviceData);
    }

    malloced = true;

}

void Glpa::Render2d::dRelease()
{
    if (!malloced) return;

    cudaFree(dBuf);
    delete hBuf;

    cudaFree(dImgPosX);
    cudaFree(dImgPosY);
    cudaFree(dImgWidth);
    cudaFree(dImgHeight);

    cudaFree(dImgData);

    malloced = false;
}

void Glpa::Render2d::run
(
        std::unordered_map<std::string, Glpa::SceneObject*>& objs,
        std::map<int, std::vector<std::string>>& drawOrder,
        LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
){
    if (!malloced) dMalloc(objs, drawOrder, bufWidth, bufHeight, bufDpi, bgColor);

    if (imgAmount != 0)
    {
        cudaMemcpy(dBuf, hBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyHostToDevice);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = imgAmount;
        int dataSizeX = maxImgWidth * maxImgHeight;

        int desiredThreadsPerBlockX = 16;
        int desiredThreadsPerBlockY = 16;

        int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
        int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

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
}

Glpa::Render3d::~Render3d()
{
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
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < imgAmount)
    {
        if (j < imgWidth[i] * imgHeight[i])
        {
            /* 
            size = width * height
            point = posX + posY * width * dpi
            buf = point + jX + jY * width * dpi

            0(0,0) 1(1,0) 2(2,0) 3(3,0) 4(4,0)
            5(0,1) 6(1,1) 7(2,1) 8(3,1) 9(4,1)

            i % width -> x coordinate
            i / width -> y coordinate
             */

            // int drawPoint = imgPosX[i] + imgPosY[i] * bufWidth * bufDpi;
            int xCoordImg = j % imgWidth[i];
            int yCoordImg = j / imgWidth[i];

            int xCoord = imgPosX[i] + xCoordImg;
            int yCoord = imgPosY[i] + yCoordImg;

            int xWriteIF = (xCoord >= 0 && xCoord < bufWidth) ? TRUE : FALSE;
            int yWriteIF = (yCoord >= 0 && yCoord < bufHeight) ? TRUE : FALSE;

            int writeIF = (xWriteIF == TRUE && yWriteIF == TRUE) ? TRUE : FALSE;

            for (int cb1 = 0; cb1 < writeIF; cb1++)
            {
                int alphaBlendIF = (buf[xCoord + yCoord * bufWidth * bufDpi] == background) ? FALSE : TRUE;

                // If the update hasn't happened yet
                buf[xCoord + yCoord * bufWidth * bufDpi]
                = (buf[xCoord + yCoord * bufWidth * bufDpi] == background)
                ? imgData[i][xCoordImg + yCoordImg * imgWidth[i]] : buf[xCoord + yCoord * bufWidth * bufDpi];

                for (int cb2 = 0; cb2 < alphaBlendIF; cb2++)
                {
                    BYTE bufA = (buf[xCoord + yCoord * bufWidth * bufDpi] >> 24) & 0xFF;
                    BYTE bufR = (buf[xCoord + yCoord * bufWidth * bufDpi] >> 16) & 0xFF;
                    BYTE bufG = (buf[xCoord + yCoord * bufWidth * bufDpi] >> 8) & 0xFF;
                    BYTE bufB = buf[xCoord + yCoord * bufWidth * bufDpi] & 0xFF;

                    BYTE imgA = (imgData[i][xCoordImg + yCoordImg * imgWidth[i]] >> 24) & 0xFF;
                    BYTE imgR = (imgData[i][xCoordImg + yCoordImg * imgWidth[i]] >> 16) & 0xFF;
                    BYTE imgG = (imgData[i][xCoordImg + yCoordImg * imgWidth[i]] >> 8) & 0xFF;
                    BYTE imgB = imgData[i][xCoordImg + yCoordImg * imgWidth[i]] & 0xFF;

                    float alpha = static_cast<float>(imgA) / 255.0f;
                    float invAlpha = 1.0f - alpha;

                    bufA = static_cast<unsigned char>(imgA + invAlpha * bufA);
                    bufR = static_cast<unsigned char>(alpha * imgR + invAlpha * bufR);
                    bufG = static_cast<unsigned char>(alpha * imgG + invAlpha * bufG);
                    bufB = static_cast<unsigned char>(alpha * imgB + invAlpha * bufB);

                    buf[xCoord + yCoord * bufWidth * bufDpi] = (bufA << 24) | (bufR << 16) | (bufG << 8) | bufB;
                }
            }
        }
    }
}

__global__ void Glpa::Gpu2dDrawBackground(LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, DWORD background)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < bufWidth)
    {
        if (j < bufHeight)
        {
            buf[i + j * bufWidth * bufDpi] = background;
        }
    }
}
