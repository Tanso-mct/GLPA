#include "matrix.cuh"


__global__ void glpaGpu4x4_4x1sMtProduct(double *mt4x4, double *mt4x1s, double *resultMt, int mt4x1sSize){
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < mt4x1sSize && j < 3){
        resultMt[i*3 + j] 
        = mt4x1s[i*3 + 0] * mt4x4[j*4 + 0]
        + mt4x1s[i*3 + 1] * mt4x4[j*4 + 1]
        + mt4x1s[i*3 + 2] * mt4x4[j*4 + 2]
        + 1 * mt4x4[j*4 + 3];
    }
}


std::vector<Vec3d> Matrix::transRotConvert(Vec3d trans, Vec3d rot, std::vector<Vec3d> sourceVecs){
    hLeftMt = (double*)malloc(sizeof(double)*4*4);
    hRightMt = (double*)malloc(sizeof(double)*sourceVecs.size()*3);
    hResultMt = (double*)malloc(sizeof(double)*sourceVecs.size()*3);

    hLeftMt[0] = cos(RAD(rot.z)) * cos(RAD(rot.y));
    hLeftMt[1] = cos(RAD(rot.z)) * sin(RAD(rot.y)) * sin(RAD(rot.x)) * -sin(RAD(rot.z)) * cos(RAD(rot.x));
    hLeftMt[2] = cos(RAD(rot.z)) * sin(RAD(rot.y)) * cos(RAD(rot.x)) * -sin(RAD(rot.z)) * -sin(RAD(rot.x));
    hLeftMt[3] = cos(RAD(rot.z)) * cos(RAD(rot.y)) * trans.x * sin(RAD(rot.y)) * sin(RAD(rot.x)) * trans.y * cos(RAD(rot.x)) * trans.z * -sin(RAD(rot.z)) * cos(RAD(rot.x)) * trans.y * -sin(RAD(rot.x)) * trans.z;
    hLeftMt[4] = sin(RAD(rot.z)) * cos(RAD(rot.y));
    hLeftMt[5] = sin(RAD(rot.z)) * sin(RAD(rot.y)) * sin(RAD(rot.x)) * cos(RAD(rot.z)) * cos(RAD(rot.x));
    hLeftMt[6] = sin(RAD(rot.z)) * sin(RAD(rot.y)) * cos(RAD(rot.x)) * cos(RAD(rot.z)) * -sin(RAD(rot.x));
    hLeftMt[7] = sin(RAD(rot.z)) * cos(RAD(rot.y)) * trans.x * sin(RAD(rot.y)) * sin(RAD(rot.x)) * trans.y * cos(RAD(rot.x)) * trans.z * cos(RAD(rot.z)) * cos(RAD(rot.x)) * trans.y * -sin(RAD(rot.x)) * trans.z;
    hLeftMt[8] = -sin(RAD(rot.y));
    hLeftMt[9] = cos(RAD(rot.y)) * sin(RAD(rot.x));
    hLeftMt[10] = cos(RAD(rot.y)) * cos(RAD(rot.x));
    hLeftMt[11] = -sin(RAD(rot.y)) * trans.x * cos(RAD(rot.y)) * sin(RAD(rot.x)) * trans.y * cos(RAD(rot.x)) * trans.z;
    hLeftMt[12] = 0;
    hLeftMt[13] = 0;
    hLeftMt[14] = 0;
    hLeftMt[15] = 1;

    memcpy(hRightMt, sourceVecs.data(), sizeof(double)*sourceVecs.size()*3);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dLeftMt, sizeof(double)*4*4);
    cudaMalloc((void**)&dRightMt, sizeof(double)*sourceVecs.size()*3);
    cudaMalloc((void**)&dResultMt, sizeof(double)*sourceVecs.size()*3);

    // Copy host-side data to device-side memory
    cudaMemcpy(dLeftMt, hLeftMt, sizeof(double)*4*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dRightMt, hRightMt, sizeof(double)*sourceVecs.size()*3, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((sourceVecs.size() + dimBlock.x - 1) 
    / dimBlock.x, (sourceVecs.size() + dimBlock.y - 1) / dimBlock.y); // Grid Size
    glpaGpu4x4_4x1sMtProduct<<<dimGrid, dimBlock>>>
    (dLeftMt, dRightMt, dResultMt, sourceVecs.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMt, dResultMt, sizeof(double)*sourceVecs.size()*3, cudaMemcpyDeviceToHost);
    
    std::vector<Vec3d> rtCalcVec(sourceVecs.size());


    return std::vector<Vec3d>();
}

