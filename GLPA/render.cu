#include "render.cuh"

__global__ void glpaGpuPrepareObj(
    int objSize,
    float* objWVs,
    float* mtCamTransRot,
    float camNearZ,
    float camFarZ,
    float* camViewAngleCos,
    int* result
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < objSize)
    {
        int objRectStatus = 0;
        float objRectOrigin[3];
        float objRectOpposite[3];
        for (int objWvsI = 0; objWvsI < 8; objWvsI++)
        {
            float vec3d[3] = {objWVs[i*8*3 + objWvsI*3 + AX], objWVs[i*8*3 + objWvsI*3 + AY], objWVs[i*8*3 + objWvsI*3 + AZ]};

            float camObjVs[3] = {
                vec3d[AX] * mtCamTransRot[0] + vec3d[AY] * mtCamTransRot[1] + vec3d[AZ] * mtCamTransRot[2] + 1 * mtCamTransRot[3],
                vec3d[AX] * mtCamTransRot[4] + vec3d[AY] * mtCamTransRot[5] + vec3d[AZ] * mtCamTransRot[6] + 1 * mtCamTransRot[7],
                vec3d[AX] * mtCamTransRot[8] + vec3d[AY] * mtCamTransRot[9] + vec3d[AZ] * mtCamTransRot[10] + 1 * mtCamTransRot[11]
            };

            int objRectStatusIF = (objRectStatus > 0) ? TRUE : FALSE;

            objRectOrigin[AX] = (objRectStatusIF == FALSE) ? camObjVs[AX] : (camObjVs[AX] < objRectOrigin[AX]) ? camObjVs[AX] : objRectOrigin[AX];
            objRectOrigin[AY] = (objRectStatusIF == FALSE) ? camObjVs[AY] : (camObjVs[AY] < objRectOrigin[AY]) ? camObjVs[AY] : objRectOrigin[AY];
            objRectOrigin[AZ] = (objRectStatusIF == FALSE) ? camObjVs[AZ] : (camObjVs[AZ] > objRectOrigin[AZ]) ? camObjVs[AZ] : objRectOrigin[AZ];

            objRectOpposite[AX] = (objRectStatusIF == FALSE) ? camObjVs[AX] : (camObjVs[AX] > objRectOpposite[AX]) ? camObjVs[AX] : objRectOpposite[AX];
            objRectOpposite[AY] = (objRectStatusIF == FALSE) ? camObjVs[AY] : (camObjVs[AY] > objRectOpposite[AY]) ? camObjVs[AY] : objRectOpposite[AY];
            objRectOpposite[AZ] = (objRectStatusIF == FALSE) ? camObjVs[AZ] : (camObjVs[AZ] < objRectOpposite[AZ]) ? camObjVs[AZ] : objRectOpposite[AZ];

            objRectStatus += 1;

        }

        float objOppositeVs[12] = {
            objRectOrigin[AX], 0, objRectOpposite[AZ],
            objRectOpposite[AX], 0, objRectOpposite[AZ],
            0, objRectOrigin[AY], objRectOpposite[AZ],
            0, objRectOpposite[AY], objRectOpposite[AZ]
        };


        float zVec[3] = {0, 0, -1};
        float vecsCos[4];

        for (int aryI = 0; aryI < 4; aryI++){
            vecsCos[aryI]
            = (zVec[AX] * objOppositeVs[aryI*3 + AX] + zVec[AY] * objOppositeVs[aryI*3 + AY] + zVec[AZ] * objOppositeVs[aryI*3 + AZ]) /
            (sqrt(zVec[AX] * zVec[AX] + zVec[AY] * zVec[AY] + zVec[AZ] * zVec[AZ]) * 
            sqrt(objOppositeVs[aryI*3 + AX] * objOppositeVs[aryI*3 + AX] + objOppositeVs[aryI*3 + AY] * objOppositeVs[aryI*3 + AY] + 
            objOppositeVs[aryI*3 + AZ] * objOppositeVs[aryI*3 + AZ]));
        }

        int objZInIF = (objRectOrigin[AZ] >= -camFarZ && objRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;
        int objXzInIF = (vecsCos[0] >= camViewAngleCos[AX] || vecsCos[1] >= camViewAngleCos[AX]) ? TRUE : FALSE;
        int objYzInIF = (vecsCos[2] >= camViewAngleCos[AY] || vecsCos[3] >= camViewAngleCos[AY]) ? TRUE : FALSE;

        int objInIF = (objZInIF == TRUE && objXzInIF == TRUE && objYzInIF == TRUE) ? i + 1 : 0;
        // int objInIF2 = (objZInIF == TRUE && objXzInIF == TRUE && objYzInIF == TRUE) ? 2 : 0;

        result[objInIF] = TRUE;
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

    mtCamTransRot = new float[16];
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

    camViewAngleCos = new float[2];
    camViewAngleCos[AX] = cam.viewAngleCos.x;
    camViewAngleCos[AY] = cam.viewAngleCos.y;

    objInJudgeAry = new int[sObj.size() + 1];
    std::fill(objInJudgeAry, objInJudgeAry + sObj.size() + 1, FALSE); 

    float* dObjWvs;
    cudaMalloc((void**)&dObjWvs, sizeof(float)*objWvsSize);
    cudaMemcpy(dObjWvs, objWvs, sizeof(float)*objWvsSize, cudaMemcpyHostToDevice);

    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*16);
    cudaMemcpy(dMtCamTransRot, mtCamTransRot, sizeof(float)*16, cudaMemcpyHostToDevice);

    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*2);
    cudaMemcpy(dCamViewAngleCos, camViewAngleCos, sizeof(float)*2, cudaMemcpyHostToDevice);

    int* dObjInJudgeAry;
    cudaMalloc((void**)&dObjInJudgeAry, sizeof(int)*(sObj.size() + 1));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSizeY = sObj.size();
    int dataSizeX = 1; // 4 because there are two opposite v's on each of the xz plane and yz plane.

    int desiredThreadsPerBlockX = 16;
    int desiredThreadsPerBlockY = 16;

    int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
    int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

    int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
    int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

    dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
    dim3 dimGrid(blocksX, blocksY);

    glpaGpuPrepareObj<<<dimGrid, dimBlock>>>(
        sObj.size(),
        dObjWvs,
        dMtCamTransRot,
        static_cast<float>(cam.nearZ) / CALC_SCALE,
        static_cast<float>(cam.farZ) / CALC_SCALE,
        dCamViewAngleCos,
        dObjInJudgeAry
    );

    cudaError_t error = cudaGetLastError();
    if (error != 0){
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }

    cudaMemcpy(objInJudgeAry, dObjInJudgeAry, sizeof(int)*(sObj.size() + 1), cudaMemcpyDeviceToHost);

    delete[] objWvs;

    cudaFree(dObjWvs);
    cudaFree(dMtCamTransRot);
    cudaFree(dCamViewAngleCos);
    cudaFree(dObjInJudgeAry);


}

void Render::render(std::unordered_map<std::wstring, Object> sObj, Camera cam, LPDWORD buffer){
    
}
