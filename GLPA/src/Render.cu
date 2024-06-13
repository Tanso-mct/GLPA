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
                    size_t dataSize = imgPos.x * imgPos.y * sizeof(DWORD);
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
            Glpa::Color instColor(255, 255, 255, 1);
            backgroundColor = instColor.GetDword();
        }
        else
        {
            Glpa::Color instColor(255, 255, 255, 1);
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

        cudaFree(dImgPosX);
        cudaFree(dImgPosY);
        cudaFree(dImgWidth);
        cudaFree(dImgHeight);
        cudaFree(dImgData);
        cudaFree(dBuf);
    }
    else
    {
        LPDWORD dBuf;

        DWORD backgroundColor;
        if (bgColor == Glpa::BACKGROUND_BLACK)
        {
            Glpa::Color instColor(255, 255, 255, 1);
            backgroundColor = instColor.GetDword();
        }
        else
        {
            Glpa::Color instColor(255, 255, 255, 1);
            backgroundColor = instColor.GetDword();
        }

        cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
        cudaMemcpy(dBuf, buf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyHostToDevice);

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

            point = x + y * width
            
             */

            int drawPoint = imgPosX[i] + imgPosY[i] * bufWidth;
            buf[drawPoint + (j % )];
        }
    }
}

__global__ void Glpa::Gpu2dDrawBackground(LPDWORD buf, int bufWidth, int bufHeight, int bufDpi, DWORD background)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
}
