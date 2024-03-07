#include "vector.cuh"

__global__ void glpaGpuGetVecsCos(
    double* leftVec,
    double* rightVecs,
    double* resultVecs,
    int rightVecsSize
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rightVecsSize){
        resultVecs[i] 
        = (leftVec[0] * rightVecs[i*3 + 0] + leftVec[1] * rightVecs[i*3 + 1] + leftVec[2] * rightVecs[i*3 + 2]) /
        (sqrt(leftVec[0] * leftVec[0] + leftVec[1] * leftVec[1] + leftVec[2] * leftVec[2]) * 
        sqrt(rightVecs[i*3 + 0] * rightVecs[i*3 + 0] + rightVecs[i*3 + 1] * rightVecs[i*3 + 1] + 
        rightVecs[i*3 + 2] * rightVecs[i*3 + 2]));
    }
}


std::vector<double> Vector::getVecsDotCos(Vec3d leftVec, std::vector<Vec3d> rightVecs){
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
    cudaMemcpy(hResult, dResult, sizeof(double)*rightVecs.size(), cudaMemcpyDeviceToHost);
    
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


__global__ void glpaGpuGetSameSizeVecsCos(
    double* leftVecs,
    double* rightVecs,
    double* resultVecs,
    int size
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        resultVecs[i] 
        = (leftVecs[i*3] * rightVecs[i*3] + leftVecs[i*3 + 1] * rightVecs[i*3 + 1] + leftVecs[i*3 + 2] * rightVecs[i*3 + 2]) /
        (sqrt(leftVecs[i*3] * leftVecs[i*3] + leftVecs[i*3 + 1] * leftVecs[i*3 + 1] + leftVecs[i*3 + 2] * leftVecs[i*3 + 2]) * 
        sqrt(rightVecs[i*3] * rightVecs[i*3] + rightVecs[i*3 + 1] * rightVecs[i*3 + 1] + 
        rightVecs[i*3 + 2] * rightVecs[i*3 + 2]));
    }
}



std::vector<double> Vector::getSameSizeVecsDotCos(std::vector<Vec3d> leftVec, std::vector<Vec3d> rightVecs){
    int size = leftVec.size() * 3;

    hLeftVec = (double*)malloc(sizeof(double)*size);
    hRightVec = (double*)malloc(sizeof(double)*size);
    hResult = (double*)malloc(sizeof(double)*size / 3);

    memcpy(hLeftVec, leftVec.data(), sizeof(double)*size);
    memcpy(hRightVec, rightVecs.data(), sizeof(double)*size);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dLeftVec, sizeof(double)*size);
    cudaMalloc((void**)&dRightVec, sizeof(double)*size);
    cudaMalloc((void**)&dResult, sizeof(double)*size / 3);

    // Copy host-side data to device-side memory
    cudaMemcpy(dLeftVec, hLeftVec, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(dRightVec, hRightVec, sizeof(double)*size, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    int blockSize = 1024;
    int numBlocks = (size / 3 + blockSize - 1) / blockSize;

    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    glpaGpuGetSameSizeVecsCos<<<dimGrid, dimBlock>>>
    (dLeftVec, dRightVec, dResult, size);

    // Copy results from device memory to host memory
    cudaMemcpy(hResult, dResult, sizeof(double)*size / 3, cudaMemcpyDeviceToHost);
    
    std::vector<double> rtCalcNum(size / 3);

    for (int i = 0; i < size / 3; i++){
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


void Vector::pushVecToDouble(std::vector<Vec3d> sourceVec, std::vector<double> *targetVec, int vecI){
    (*targetVec).push_back(sourceVec[vecI].x);
    (*targetVec).push_back(sourceVec[vecI].y);
    (*targetVec).push_back(sourceVec[vecI].z);
}

bool Vector::ascending(double a, double b){
    return a > b;
}


bool Vector::descending(double a, double b){
    return a < b;
}


std::vector<int> Vector::sortDecenOrder(std::vector<double>& sourceNums){
    std::vector<int> rtIs(sourceNums.size());

    if (sourceNums.size() == 0){
        return rtIs;
    }
    
    std::vector<double> beforeNums = sourceNums;

    std::sort(sourceNums.begin(), sourceNums.end(), [this](double a, double b){
        return this->descending(a, b);
    });

    std::vector<double>::iterator itDouble;
    int index;
    for (int i = 0; i < sourceNums.size(); i++){
        itDouble = std::find(beforeNums.begin(), beforeNums.end(), sourceNums[i]);
        index = std::distance(beforeNums.begin(), itDouble);
        rtIs[i] = index;
    }

    return rtIs;
}


std::vector<int> Vector::sortAsenOrder(std::vector<double>& sourceNums){
    std::vector<int> rtIs(sourceNums.size());

    if (sourceNums.size() == 0){
        return rtIs;
    }
    
    std::vector<double> beforeNums = sourceNums;

    std::sort(sourceNums.begin(), sourceNums.end(), [this](double a, double b){
        return this->ascending(a, b);
    });

    std::vector<double>::iterator itDouble;
    int index;
    for (int i = 0; i < sourceNums.size(); i++){
        itDouble = std::find(beforeNums.begin(), beforeNums.end(), sourceNums[i]);
        index = std::distance(beforeNums.begin(), itDouble);
        rtIs[i] = index;
    }

    return rtIs;
}
