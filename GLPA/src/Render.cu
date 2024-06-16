#include "Render.cuh"

Glpa::Render2d::Render2d()
{
}

Glpa::Render2d::~Render2d()
{
}

void Glpa::Render2d::run
(
    std::unordered_map<std::string, Glpa::SceneObject*> objs,
    std::map<int, std::vector<std::string>> drawOrder,
    HDC dc, LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, std::string bgColor
){
    // i = image amount
    // j = this image width x height

    // Separate processing depending on image or text.

    int totalImg = 0;
    int totalText = 0;

    std::vector<int> hImgPosX;
    std::vector<int> hImgPosY;
    std::vector<int> hImgWidth;
    std::vector<int> hImgHeight;
    std::vector<LPDWORD> hImgData;

    int maxImgWidth = 0;
    int maxImgHeight = 0;

    for (auto& pair : drawOrder)
    {
        for (int i = 0; i < pair.second.size(); i++)
        {
            if (Glpa::Image* img = dynamic_cast<Glpa::Image*>(objs[pair.second[i]]))
            {
                if (img->getVisible())
                {
                    Vec2d imgPos = img->getPos();
                    hImgPosX.push_back(imgPos.x);
                    hImgPosY.push_back(imgPos.y);
                    hImgWidth.push_back(img->getWidth());
                    hImgHeight.push_back(img->getHeight());

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

    if (hImgData.size() != 0)
    {
        int* dImgPosX;
        int* dImgPosY;
        int* dImgWidth;
        int* dImgHeight;
        LPDWORD* dImgData;
        LPDWORD dBuf;

        DWORD backgroundColor;
        if (bgColor == Glpa::BACKGROUND_BLACK)
        {
            Glpa::Color instColor(0, 200, 0, 1);
            backgroundColor = instColor.GetDword();
        }
        else
        {
            Glpa::Color instColor(0, 200, 0, 1);
            backgroundColor = instColor.GetDword();
        }

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

        std::memset(buf, 0, sizeof(bufWidth * bufHeight * bufDpi));
        cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
        cudaMemcpy(dBuf, buf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyHostToDevice);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = hImgData.size();
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
            dImgPosX, dImgPosY, dImgWidth, dImgHeight, dImgData, hImgData.size(), 
            dBuf, bufWidth, bufHeight, bufDpi, backgroundColor
        );
        cudaError_t error = cudaGetLastError();
        if (error != 0){
            OutputDebugStringA("GlpaLib ERROR Render.cu - Processing with Cuda failed.\n");
            throw std::runtime_error("Processing with Cuda failed.");
        }

        cudaMemcpy(buf, dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyDeviceToHost);

        cudaFree(dImgPosX);
        cudaFree(dImgPosY);
        cudaFree(dImgWidth);
        cudaFree(dImgHeight);

        for (int i = 0; i < hImgData.size(); i++)
        {
            DWORD* ptDeviceData;
        
            cudaMemcpy(&ptDeviceData, &dImgData[i], sizeof(DWORD*), cudaMemcpyDeviceToHost);

            cudaFree(ptDeviceData);
        }

        cudaFree(dImgData);

        cudaFree(dBuf);
    }
    else
    {
        LPDWORD dBuf;

        DWORD backgroundColor;
        if (bgColor == Glpa::BACKGROUND_BLACK)
        {
            Glpa::Color instColor(0, 0, 0, 1);
            backgroundColor = instColor.GetDword();
        }
        else
        {
            Glpa::Color instColor(0, 200, 0, 1);
            backgroundColor = instColor.GetDword();
        }

        cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = bufWidth;
        int dataSizeX = bufHeight;

        int desiredThreadsPerBlockX = 16;
        int desiredThreadsPerBlockY = 16;

        int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
        int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

        int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
        int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

        dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
        dim3 dimGrid(blocksX, blocksY);

        Gpu2dDrawBackground<<<dimGrid, dimBlock>>>(dBuf, bufWidth, bufHeight, bufDpi, backgroundColor);
        cudaError_t error = cudaGetLastError();
        if (error != 0){
            OutputDebugStringA("GlpaLib ERROR Render.cu - Processing with Cuda failed.\n");
            throw std::runtime_error("Processing with Cuda failed.");
        }

        cudaMemcpy(buf, dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyDeviceToHost);

        cudaFree(dBuf);

    }

}

Glpa::Render3d::Render3d()
{
}

Glpa::Render3d::~Render3d()
{
}

void Glpa::Render3d::run(std::unordered_map<std::string, Glpa::SceneObject*> objs, HDC dc, LPDWORD buf)
{

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

            int drawPoint = imgPosX[i] + imgPosY[i] * bufWidth * bufDpi;
            int xCoord = j % imgWidth[i];
            int yCoord = j / imgWidth[i];

            int alphaBlendIF = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] == 0) ? FALSE : TRUE;
            alphaBlendIF = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] == background) ? FALSE : TRUE;

            // If initialization is required
            buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] 
            = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] == 0) 
            ? background : buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi];

            // If the update hasn't happened yet
            buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi]
            = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] == background)
            ? imgData[i][imgPosX[i] + imgPosY[i] * imgWidth[i]] : buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi];

            for (int fI = 0; fI < alphaBlendIF; fI++)
            {
                BYTE bufA = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] >> 24) & 0xFF;
                BYTE bufR = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] >> 16) & 0xFF;
                BYTE bufG = (buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] >> 8) & 0xFF;
                BYTE bufB = buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] & 0xFF;

                BYTE imgA = (imgData[i][imgPosX[i] + imgPosY[i] * imgWidth[i]] >> 24) & 0xFF;
                BYTE imgR = (imgData[i][imgPosX[i] + imgPosY[i] * imgWidth[i]] >> 16) & 0xFF;
                BYTE imgG = (imgData[i][imgPosX[i] + imgPosY[i] * imgWidth[i]] >> 8) & 0xFF;
                BYTE imgB = imgData[i][imgPosX[i] + imgPosY[i] * imgWidth[i]] & 0xFF;

                float alpha = static_cast<float>(imgA) / 255.0f;
                float invAlpha = 1.0f - alpha;

                bufA = static_cast<unsigned char>(imgA + invAlpha * bufA);
                bufR = static_cast<unsigned char>(alpha * imgR + invAlpha * bufR);
                bufG = static_cast<unsigned char>(alpha * imgG + invAlpha * bufG);
                bufB = static_cast<unsigned char>(alpha * imgB + invAlpha * bufB);

                buf[drawPoint + xCoord + yCoord * bufWidth * bufDpi] = (bufA << 24) | (bufR << 16) | (bufG << 8) | bufB;
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
