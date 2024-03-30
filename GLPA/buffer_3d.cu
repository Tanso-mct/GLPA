#include "buffer_3d.cuh"


void Buffer3d::initialize(Vec2d argScPixelSize, int argScDpi){
    scPixelSize = argScPixelSize;
    scDpi = argScDpi;
}


__global__ void glpaGpuDrawZBuff(
    double* dZbComp,
    LPDWORD dWndBuff,
    int scPixelSizeX,
    int scPixelSizeY,
    int scDpi
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < scPixelSizeY){
        if (j < scPixelSizeX){
            dWndBuff[j + i*scPixelSizeX * scDpi]
            = (1 << 24) | 
            (static_cast<int>(255 * dZbComp[j + i*(scPixelSizeX + 1)]) << 16) | 
            (static_cast<int>(255 * dZbComp[j + i*(scPixelSizeX + 1)]) << 8) | 
            static_cast<int>(255 * dZbComp[j + i*(scPixelSizeX + 1)]);
        }
    }


}


void Buffer3d::drawZBuff(LPDWORD wndBuff, double* &zbComp){
    int sizeWndBuff = scPixelSize.x * scPixelSize.y;
    int sizeZbComp = (scPixelSize.x + 1) * (scPixelSize.y + 1);

    LPDWORD dWndBuff;
    double* dZbComp;
    cudaMalloc((void**)&dWndBuff, sizeof(DWORD)*sizeWndBuff);
    cudaMalloc((void**)&dZbComp, sizeof(double)*sizeZbComp);

    cudaMemcpy(dZbComp, zbComp, sizeof(double)*sizeZbComp, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSizeY = scPixelSize.y;
    int dataSizeX = scPixelSize.x;
    int desiredThreadsPerBlockX = 16;
    int desiredThreadsPerBlockY = 16;

    int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
    int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

    int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
    int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

    dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 dimGrid(blocksX, blocksY);

    glpaGpuDrawZBuff<<<dimGrid, dimBlock>>>(
        dZbComp,
        dWndBuff,
        scPixelSize.x,
        scPixelSize.y,
        scDpi
    );

    cudaError_t error = cudaGetLastError();
    if (error != 0){
        throw std::runtime_error(ERROR_CAMERA_CUDA_ERROR);
    }

    cudaMemcpy(wndBuff, dWndBuff, sizeof(DWORD)*sizeWndBuff, cudaMemcpyDeviceToHost);

    cudaFree(dZbComp);
    cudaFree(dWndBuff);


}
