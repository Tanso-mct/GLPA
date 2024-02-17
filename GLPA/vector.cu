#include "vector.cuh"

__global__ void glpaGpuGetVecsCos(
    double* leftVec,
    double* rightVecs,
    double* resultVecs,
    int rightVecsSize
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rightVecsSize){
        rightVecs[i] 
        = leftVec[0] * rightVecs[i*3 + 0] + leftVec[1] * rightVecs[i*3 + 1] + leftVec[2] * rightVecs[i*3 + 2] /
        (sqrt(leftVec[0] * leftVec[0] + leftVec[1] * leftVec[1] + leftVec[2] * leftVec[2]) * 
        sqrt(rightVecs[i*3 + 0] * rightVecs[i*3 + 0] + rightVecs[i*3 + 1] * rightVecs[i*3 + 1] + 
        rightVecs[i*3 + 2] * rightVecs[i*3 + 2]));
    }
}


std::vector<double> Vector::getVecsCis(Vec3d leftVec, std::vector<Vec3d> rightVecs)
{
    hLeftVec = (double*)malloc(sizeof(double)*3);
    hRightVec = (double*)malloc(sizeof(double)*rightVecs.size()*3);
    hResult = (double*)malloc(sizeof(double)*rightVecs.size());

    hLeftVec[0] = leftVec.x;
    hLeftVec[1] = leftVec.y;
    hLeftVec[2] = leftVec.z;
    memcpy(hRightVec, rightVecs.data(), sizeof(double)*rightVecs.size()*3);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dLeftVec, sizeof(double)*3);
    cudaMalloc((void**)&dRightVec, sizeof(double)*rightVecs.size()*3);
    cudaMalloc((void**)&dResult, sizeof(double)*rightVecs.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dLeftVec, hLeftVec, sizeof(double)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dRightVec, hRightVec, sizeof(double)*rightVecs.size()*3, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    int blockSize = 1024;
    int numBlocks = (rightVecs.size() + blockSize - 1) / blockSize;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    glpaGpuGetVecsCos<<<dimGrid, dimBlock>>>
    (dLeftVec, dRightVec, dResult, rightVecs.size());

    // Copy results from device memory to host memory
    cudaMemcpy(dResult, hResult, sizeof(double)*rightVecs.size(), cudaMemcpyDeviceToHost);
    
    std::vector<double> rtCalcNum(rightVecs.size());

    for (int i = 0; i < rightVecs.size(); i++){
        rtCalcNum[i] = hResult[i];
    }

    // Release all memory allocated by malloc
    free(hLeftVec);
    free(hRightVec);
    free(hResult);

    cudaFree(dLeftVec);
    cudaFree(dRightVec);
    cudaFree(dResult);

    return rtCalcNum;
}
