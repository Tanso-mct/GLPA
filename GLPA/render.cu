#include "render.cuh"

__global__ void glpaGpuPreparePoly(
    int objSize,
    double* objWVs
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < objSize){
        float* transRotMat = new float[16];
        transRotMat[0] = cos(RAD(-rot.z)) * cos(RAD(-rot.y));
        transRotMat[1] = cos(RAD(-rot.z)) * sin(RAD(-rot.y)) * sin(RAD(-rot.x)) + -sin(RAD(-rot.z)) * cos(RAD(-rot.x));
        transRotMat[2] = cos(RAD(-rot.z)) * sin(RAD(-rot.y)) * cos(RAD(-rot.x)) + -sin(RAD(-rot.z)) * -sin(RAD(-rot.x));
        transRotMat[3] = -trans.x;
        transRotMat[4] = sin(RAD(-rot.z)) * cos(RAD(-rot.y));
        transRotMat[5] = sin(RAD(-rot.z)) * sin(RAD(-rot.y)) * sin(RAD(-rot.x)) + cos(RAD(-rot.z)) * cos(RAD(-rot.x));
        transRotMat[6] = sin(RAD(-rot.z)) * sin(RAD(-rot.y)) * cos(RAD(-rot.x)) + cos(RAD(-rot.z)) * -sin(RAD(-rot.x));
        transRotMat[7] = -trans.y;
        transRotMat[8] = -sin(RAD(-rot.y));
        transRotMat[9] = cos(RAD(-rot.y)) * sin(RAD(-rot.x));
        transRotMat[10] = cos(RAD(-rot.y)) * cos(RAD(-rot.x));
        transRotMat[11] = -trans.z;
        transRotMat[12] = 0;
        transRotMat[13] = 0;
        transRotMat[14] = 0;
        transRotMat[15] = 1;
    }
}

void Render::prepareObjs(std::unordered_map<std::wstring, Object> sObj, Camera cam){
    int sObjSize = sObj.size();
    int objWvsSize = sObjSize*8*3;

    double* objWvs = new double[objWvsSize];

    int roopObj = 0;
    for (auto obj : sObj){
        for (int i = 0; i < 8; i++){
            objWvs[roopObj*8*3 + i*3] = obj.second.range.wVertex[i].x;
            objWvs[roopObj*8*3 + i*3 + 1] = obj.second.range.wVertex[i].y;
            objWvs[roopObj*8*3 + i*3 + 2] = obj.second.range.wVertex[i].z;
        }
        roopObj += 1;
    }


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSizeY = sObj.size();
    int dataSizeX = 4; // 4 because there are two opposite v's on each of the xz plane and yz plane.

    int desiredThreadsPerBlockX = 16;
    int desiredThreadsPerBlockY = 16;

    int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
    int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

    int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
    int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

    dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 dimGrid(blocksX, blocksY);

    glpaGpuPreparePoly<<<dimGrid, dimBlock>>>(

    );
}

void Render::render(std::unordered_map<std::wstring, Object> sObj, Camera cam, LPDWORD buffer){
    
}
