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
    std::vector<int> hImgSize;
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
                    hImgSize.push_back(img->getWidth() * img->getHeight());

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
        int* dImgSize;
        LPDWORD* dImgData;
        LPDWORD dBuf;

        int* dBufWidth;
        int* dBufHeight;
        int* dBufDpi;

        cudaMalloc(&dImgPosX, hImgPosX.size() * sizeof(int));
        cudaMemcpy(dImgPosX, hImgPosX.data(), hImgPosX.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dImgPosY, hImgPosY.size() * sizeof(int));
        cudaMemcpy(dImgPosY, hImgPosY.data(), hImgPosY.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dImgSize, hImgSize.size() * sizeof(int));
        cudaMemcpy(dImgSize, hImgSize.data(), hImgSize.size() * sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dImgData, hImgData.size() * sizeof(DWORD*));
        cudaMemcpy(dImgData, hImgData.data(), hImgData.size() * sizeof(DWORD*), cudaMemcpyHostToDevice);

        cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
        cudaMemcpy(dBuf, buf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufWidth, sizeof(int));
        cudaMemcpy(dBufWidth, &bufWidth, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufHeight, sizeof(int));
        cudaMemcpy(dBufHeight, &bufHeight, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufDpi, sizeof(int));
        cudaMemcpy(dBufDpi, &bufDpi, sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = maxImgWidth;
        int dataSizeX = maxImgHeight;

        int desiredThreadsPerBlockX = 16;
        int desiredThreadsPerBlockY = 16;

        int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
        int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

        int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
        int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

        dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
        dim3 dimGrid(blocksX, blocksY);
    }
    else
    {
        LPDWORD dBuf;

        int* dBufWidth;
        int* dBufHeight;
        int* dBufDpi;

        cudaMalloc(&dBuf, bufWidth * bufHeight * bufDpi * sizeof(DWORD));
        cudaMemcpy(dBuf, buf, bufWidth * bufHeight * bufDpi * sizeof(DWORD), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufWidth, sizeof(int));
        cudaMemcpy(dBufWidth, &bufWidth, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufHeight, sizeof(int));
        cudaMemcpy(dBufHeight, &bufHeight, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc(&dBufDpi, sizeof(int));
        cudaMemcpy(dBufDpi, &bufDpi, sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int dataSizeY = maxImgWidth;
        int dataSizeX = maxImgHeight;

        int desiredThreadsPerBlockX = 16;
        int desiredThreadsPerBlockY = 16;

        int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
        int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

        int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
        int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

        dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
        dim3 dimGrid(blocksX, blocksY);
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
