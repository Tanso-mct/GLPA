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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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

        for (int aryI = 0; aryI < 4; aryI++)
        {
            float calcObjOppositeV[3] = {
                objOppositeVs[aryI*3 + AX],
                objOppositeVs[aryI*3 + AY],
                objOppositeVs[aryI*3 + AZ]
            };

            vecGetVecsCos(zVec, calcObjOppositeV, &vecsCos[aryI]);

            // vecsCos[aryI]
            // = (zVec[AX] * objOppositeVs[aryI*3 + AX] + zVec[AY] * objOppositeVs[aryI*3 + AY] + zVec[AZ] * objOppositeVs[aryI*3 + AZ]) /
            // (sqrt(zVec[AX] * zVec[AX] + zVec[AY] * zVec[AY] + zVec[AZ] * zVec[AZ]) * 
            // sqrt(objOppositeVs[aryI*3 + AX] * objOppositeVs[aryI*3 + AX] + objOppositeVs[aryI*3 + AY] * objOppositeVs[aryI*3 + AY] + 
            // objOppositeVs[aryI*3 + AZ] * objOppositeVs[aryI*3 + AZ]));
        }

        int objZInIF = (objRectOrigin[AZ] >= -camFarZ && objRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;
        int objXzInIF = (vecsCos[0] >= camViewAngleCos[AX] || vecsCos[1] >= camViewAngleCos[AX]) ? TRUE : FALSE;
        int objYzInIF = (vecsCos[2] >= camViewAngleCos[AY] || vecsCos[3] >= camViewAngleCos[AY]) ? TRUE : FALSE;

        int objInIF = (objZInIF == TRUE && objXzInIF == TRUE && objYzInIF == TRUE) ? i + 1 : 0;

        result[objInIF] = TRUE;
    }
}

void Render::prepareObjs(std::unordered_map<std::wstring, Object> sObj, Camera cam)
{
    int sObjSize = sObj.size();
    int objWvsSize = sObjSize*8*3;

    float* hObjWvs = new float[objWvsSize];

    int roopObj = 0;
    for (auto obj : sObj)
    {
        for (int i = 0; i < 8; i++)
        {
            hObjWvs[roopObj*8*3 + i*3] = obj.second.range.wVertex[i].x / CALC_SCALE;
            hObjWvs[roopObj*8*3 + i*3 + 1] = obj.second.range.wVertex[i].y / CALC_SCALE;
            hObjWvs[roopObj*8*3 + i*3 + 2] = obj.second.range.wVertex[i].z / CALC_SCALE;
        }
        roopObj += 1;
    }

    hMtCamTransRot = new float[16];
    hMtCamTransRot[0] = cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamTransRot[1] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[2] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[3] = -cam.wPos.x / CALC_SCALE;
    hMtCamTransRot[4] = sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamTransRot[5] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[6] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[7] = -cam.wPos.y / CALC_SCALE;
    hMtCamTransRot[8] = -sin(RAD(-cam.rotAngle.y));
    hMtCamTransRot[9] = cos(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x));
    hMtCamTransRot[10] = cos(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x));
    hMtCamTransRot[11] = -cam.wPos.z / CALC_SCALE;
    hMtCamTransRot[12] = 0;
    hMtCamTransRot[13] = 0;
    hMtCamTransRot[14] = 0;
    hMtCamTransRot[15] = 1;

    hCamViewAngleCos = new float[2];
    hCamViewAngleCos[AX] = cam.viewAngleCos.x;
    hCamViewAngleCos[AY] = cam.viewAngleCos.y;

    hObjInJudgeAry = new int[sObj.size() + 1];
    std::fill(hObjInJudgeAry, hObjInJudgeAry + sObj.size() + 1, FALSE); 

    float* dObjWvs;
    cudaMalloc((void**)&dObjWvs, sizeof(float)*objWvsSize);
    cudaMemcpy(dObjWvs, hObjWvs, sizeof(float)*objWvsSize, cudaMemcpyHostToDevice);

    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*16);
    cudaMemcpy(dMtCamTransRot, hMtCamTransRot, sizeof(float)*16, cudaMemcpyHostToDevice);

    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*2);
    cudaMemcpy(dCamViewAngleCos, hCamViewAngleCos, sizeof(float)*2, cudaMemcpyHostToDevice);

    int* dObjInJudgeAry;
    cudaMalloc((void**)&dObjInJudgeAry, sizeof(int)*(sObj.size() + 1));

    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);

    // int dataSizeY = sObj.size();
    // int dataSizeX = 1; // 4 because there are two opposite v's on each of the xz plane and yz plane.

    // int desiredThreadsPerBlockX = 16;
    // int desiredThreadsPerBlockY = 16;

    // int blocksX = (dataSizeX + desiredThreadsPerBlockX - 1) / desiredThreadsPerBlockX;
    // int blocksY = (dataSizeY + desiredThreadsPerBlockY - 1) / desiredThreadsPerBlockY;

    // int threadsPerBlockX = min(desiredThreadsPerBlockX, deviceProp.maxThreadsDim[0]);
    // int threadsPerBlockY = min(desiredThreadsPerBlockY, deviceProp.maxThreadsDim[1]);

    // dim3 dimBlock(threadsPerBlockX, threadsPerBlockY);
    // dim3 dimGrid(blocksX, blocksY);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = sObj.size();
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    glpaGpuPrepareObj<<<dimGrid, dimBlock>>>
    (
        sObj.size(),
        dObjWvs,
        dMtCamTransRot,
        static_cast<float>(cam.nearZ) / CALC_SCALE,
        static_cast<float>(cam.farZ) / CALC_SCALE,
        dCamViewAngleCos,
        dObjInJudgeAry
    );

    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }

    cudaMemcpy(hObjInJudgeAry, dObjInJudgeAry, sizeof(int)*(sObj.size() + 1), cudaMemcpyDeviceToHost);

    delete[] hObjWvs;

    cudaFree(dObjWvs);
    cudaFree(dMtCamTransRot);
    cudaFree(dCamViewAngleCos);
    cudaFree(dObjInJudgeAry);


}

__device__ int judgePolyVInViewVolume(
    float* cnvtPolyV,
    float camFarZ,
    float camNearZ,
    float* camViewAngle
){
    float cnvtXzPolyV[3] = {cnvtPolyV[AX], 0, cnvtPolyV[AZ]};
    float cnvtYzPolyV[3] = {0, cnvtPolyV[AY], cnvtPolyV[AZ]};

    float zVec[3] = {0, 0, -1};

    float cnvtXzPolyVxZVecDotCos;
    vecGetVecsCos(zVec, cnvtXzPolyV, &cnvtXzPolyVxZVecDotCos);

    float cnvtYzPolyVxZVecDotCos;
    vecGetVecsCos(zVec, cnvtYzPolyV, &cnvtYzPolyVxZVecDotCos);

    int polyVZInIF = (cnvtPolyV[AZ] >= -camFarZ && cnvtPolyV[AZ] <= -camNearZ) ? TRUE : FALSE;
    int polyXzVInIF = (cnvtXzPolyVxZVecDotCos >= camViewAngle[AX]) ? TRUE : FALSE;
    int polyYzVInIF = (cnvtYzPolyVxZVecDotCos >= camViewAngle[AY]) ? TRUE : FALSE;

    return (polyVZInIF == TRUE && polyXzVInIF == TRUE && polyYzVInIF == TRUE) ? TRUE : FALSE;
}

__global__ void glpaGpuRender(
    float* polyVs,
    float* polyNs,
    int polyAmount,
    float* mtCamTransRot,
    float* mtCamRot,
    float camFarZ,
    float camNearZ,
    float* camViewAngle
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyAmount)
    {
        float vec3d[3];

        vec3d[AX] = polyVs[i*9 + AX];
        vec3d[AY] = polyVs[i*9 + AY];
        vec3d[AZ] = polyVs[i*9 + AZ];
        float cnvtPolyV1[3];
        mtProduct4x4Vec3d(mtCamTransRot, vec3d, cnvtPolyV1);

        vec3d[AX] = polyVs[i*9 + 3 + AX];
        vec3d[AY] = polyVs[i*9 + 3 + AY];
        vec3d[AZ] = polyVs[i*9 + 3 + AZ];
        float cnvtPolyV2[3];
        mtProduct4x4Vec3d(mtCamTransRot, vec3d, cnvtPolyV2);

        vec3d[AX] = polyVs[i*9 + 6 + AX];
        vec3d[AY] = polyVs[i*9 + 6 + AY];
        vec3d[AZ] = polyVs[i*9 + 6 + AZ];
        float cnvtPolyV3[3];
        mtProduct4x4Vec3d(mtCamTransRot, vec3d, cnvtPolyV3);

        vec3d[AX] = polyNs[i*3 + AX];
        vec3d[AY] = polyNs[i*3 + AY];
        vec3d[AZ] = polyNs[i*3 + AZ];
        float cnvtPolyN[3];
        mtProduct4x4Vec3d(mtCamRot, vec3d, cnvtPolyN);

        float polyVxPolyNDotCos;
        vecGetVecsCos(cnvtPolyN, cnvtPolyV1, &polyVxPolyNDotCos);
        
        int polyBilateralIF = (polyVxPolyNDotCos <= 0) ? TRUE : FALSE;

        for (int conditionalBranch = 0; conditionalBranch < polyBilateralIF; conditionalBranch++)
        {
            int polyV1InIF = judgePolyVInViewVolume(cnvtPolyV1, camFarZ, camNearZ, camViewAngle);
            int polyV2InIF = judgePolyVInViewVolume(cnvtPolyV1, camFarZ, camNearZ, camViewAngle);
            int polyV3InIF = judgePolyVInViewVolume(cnvtPolyV1, camFarZ, camNearZ, camViewAngle);

            int noVsInIF = (polyV1InIF == FALSE && polyV2InIF == FALSE && polyV3InIF == FALSE) ? TRUE : FALSE;

            for (int conditionalBranch2 = 0; conditionalBranch2 < noVsInIF; conditionalBranch2++)
            {
                
            }



        }

        


    }

}

void Render::render(std::unordered_map<std::wstring, Object> sObj, Camera cam, LPDWORD buffer)
{
    std::vector<float> polyVs;
    std::vector<float> polyNs;
    int loopObjI = 1;
    for (auto obj : sObj)
    {
        if (hObjInJudgeAry[loopObjI] == FALSE)
        {
            continue;
        }

        for (int i = 0; i < obj.second.poly.vId.size(); i++)
        {
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n1].z / CALC_SCALE);

            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n2].z / CALC_SCALE);

            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].x / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].y / CALC_SCALE);
            polyVs.push_back(obj.second.v.world[obj.second.poly.vId[i].n3].z / CALC_SCALE);

            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].x);
            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].y);
            polyNs.push_back(obj.second.v.normal[obj.second.poly.normalId[i].n1].z);
        }
    }

    hMtCamRot = new float[16];
    hMtCamRot[0] = cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamRot[1] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[2] = cos(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + -sin(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamRot[3] = 0;
    hMtCamRot[4] = sin(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.y));
    hMtCamRot[5] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[6] = sin(RAD(-cam.rotAngle.z)) * sin(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x)) + cos(RAD(-cam.rotAngle.z)) * -sin(RAD(-cam.rotAngle.x));
    hMtCamRot[7] = 0;
    hMtCamRot[8] = -sin(RAD(-cam.rotAngle.y));
    hMtCamRot[9] = cos(RAD(-cam.rotAngle.y)) * sin(RAD(-cam.rotAngle.x));
    hMtCamRot[10] = cos(RAD(-cam.rotAngle.y)) * cos(RAD(-cam.rotAngle.x));
    hMtCamRot[11] = 0;
    hMtCamRot[12] = 0;
    hMtCamRot[13] = 0;
    hMtCamRot[14] = 0;
    hMtCamRot[15] = 1;

    float* dPolyVs;
    float* dPolyNs;
    cudaMalloc((void**)&dPolyVs, sizeof(int)*polyVs.size());
    cudaMalloc((void**)&dPolyNs, sizeof(int)*polyNs.size());
    cudaMemcpy(dPolyVs, polyVs.data(), sizeof(float)*polyVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyVs, polyNs.data(), sizeof(float)*polyNs.size(), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = polyVs.size() / 9;
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    glpaGpuRender<<<dimGrid, dimBlock>>>
    (
        
    );

    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }






}
