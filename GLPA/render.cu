#include "render.cuh"

__global__ void glpaGpuPreparePoly(
    int objSize,
    float* objWVs,
    float* mtCamTransRot
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < objSize)
    {
        float vec3d[3] = {objWVs[i*8*3 + AX], objWVs[i*8*3 + AY], objWVs[i*8*3 + AZ]};

        float camObjVs[3] = {
            vec3d[AX] * mtCamTransRot[0] + vec3d[AY] * mtCamTransRot[1] + vec3d[AZ] * mtCamTransRot[2] + 1 * mtCamTransRot[3],
            vec3d[AX] * mtCamTransRot[4] + vec3d[AY] * mtCamTransRot[5] + vec3d[AZ] * mtCamTransRot[6] + 1 * mtCamTransRot[7],
            vec3d[AX] * mtCamTransRot[8] + vec3d[AY] * mtCamTransRot[9] + vec3d[AZ] * mtCamTransRot[10] + 1 * mtCamTransRot[11]
        };
    }
}

void Render::prepareObjs(std::unordered_map<std::wstring, Object> sObj, Camera cam)
{
    int sObjSize = sObj.size();
    int objWvsSize = sObjSize*8*3;

    float* objWvs = new float[objWvsSize];

    int roopObj = 0;
    for (auto obj : sObj){
        for (int i = 0; i < 8; i++){
            objWvs[roopObj*8*3 + i*3] = obj.second.range.wVertex[i].x / CALC_SCALE;
            objWvs[roopObj*8*3 + i*3 + 1] = obj.second.range.wVertex[i].y / CALC_SCALE;
            objWvs[roopObj*8*3 + i*3 + 2] = obj.second.range.wVertex[i].z / CALC_SCALE;
        }
        roopObj += 1;
    }

    float* mtCamTransRot = new float[16];
    mtCamTransRot[0] = cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    mtCamTransRot[1] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    mtCamTransRot[2] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    mtCamTransRot[3] = -cam.wPos.x / CALC_SCALE;
    mtCamTransRot[4] = sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    mtCamTransRot[5] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    mtCamTransRot[6] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    mtCamTransRot[7] = -cam.wPos.y / CALC_SCALE;
    mtCamTransRot[8] = -sin(RAD(-cam.rotAngle.y));
    mtCamTransRot[9] = cos(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x));
    mtCamTransRot[10] = cos(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x));
    mtCamTransRot[11] = -cam.wPos.z / CALC_SCALE;
    mtCamTransRot[12] = 0;
    mtCamTransRot[13] = 0;
    mtCamTransRot[14] = 0;
    mtCamTransRot[15] = 1;


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
