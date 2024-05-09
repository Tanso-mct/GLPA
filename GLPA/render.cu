#include "render.cuh"

Render::Render()
{
    hMtCamTransRot = std::vector<float>(16);
    hMtCamRot = std::vector<float>(16);

    hCamViewAngleCos = std::vector<float>(2);

}

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

            VEC_GET_VECS_COS(zVec, calcObjOppositeV, vecsCos[aryI]);
        }

        int objZInIF = (objRectOrigin[AZ] >= -camFarZ && objRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;

        // True if positive, false if negative.
        int xzOriginSymbol = (objRectOrigin[AX] >= 0) ? TRUE : FALSE;
        int xzOppositeSymbol = (objRectOpposite[AX] >= 0) ? TRUE : FALSE;
        int yzOriginSymbol = (objRectOrigin[AY] >= 0) ? TRUE : FALSE;
        int yzOppositeSymbol = (objRectOpposite[AY] >= 0) ? TRUE : FALSE;

        int objXzInIF = 
        (
            (xzOriginSymbol == TRUE && vecsCos[0] >= camViewAngleCos[AX]) || 
            (xzOriginSymbol == FALSE && xzOppositeSymbol == TRUE) ||
            (xzOppositeSymbol == TRUE && vecsCos[1] >= camViewAngleCos[AX])
        ) ? TRUE : FALSE;

        int objYzInIF = 
        (
            (yzOriginSymbol == TRUE && vecsCos[2] >= camViewAngleCos[AY]) || 
            (yzOriginSymbol == FALSE && yzOppositeSymbol == TRUE) || 
            (xzOppositeSymbol == TRUE && vecsCos[3] >= camViewAngleCos[AY])
        ) ? TRUE : FALSE;

        int objInIF = (objZInIF == TRUE && objXzInIF == TRUE && objYzInIF == TRUE) ? i + 1 : 0;

        result[objInIF] = TRUE;
    }
}

void Render::prepareObjs(std::unordered_map<std::wstring, Object> sObj, Camera cam)
{
    std::vector<float> hObjWvs;

    for (auto obj : sObj)
    {
        for (int i = 0; i < 8; i++)
        {
            hObjWvs.push_back(obj.second.range.wVertex[i].x / CALC_SCALE);
            hObjWvs.push_back(obj.second.range.wVertex[i].y / CALC_SCALE);
            hObjWvs.push_back(obj.second.range.wVertex[i].z / CALC_SCALE);
        }
    }

    float* dObjWvs;
    cudaMalloc((void**)&dObjWvs, sizeof(float)*hObjWvs.size());
    cudaMemcpy(dObjWvs, hObjWvs.data(), sizeof(float)*hObjWvs.size(), cudaMemcpyHostToDevice);

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

    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*hMtCamTransRot.size());
    cudaMemcpy(dMtCamTransRot, hMtCamTransRot.data(), sizeof(float)*hMtCamTransRot.size(), cudaMemcpyHostToDevice);

    hCamViewAngleCos[AX] = cam.viewAngleCos.x;
    hCamViewAngleCos[AY] = cam.viewAngleCos.y;

    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*hCamViewAngleCos.size());
    cudaMemcpy(dCamViewAngleCos, hCamViewAngleCos.data(), sizeof(float)*hCamViewAngleCos.size(), cudaMemcpyHostToDevice);


    hObjInJudgeAry = new int[sObj.size() + 1];
    std::fill(hObjInJudgeAry, hObjInJudgeAry + sObj.size() + 1, FALSE); 

    int* dObjInJudgeAry;
    cudaMalloc((void**)&dObjInJudgeAry, sizeof(int)*(sObj.size() + 1));


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
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }

    cudaMemcpy(hObjInJudgeAry, dObjInJudgeAry, sizeof(int)*(sObj.size() + 1), cudaMemcpyDeviceToHost);

    cudaFree(dObjWvs);
    cudaFree(dMtCamTransRot);
    cudaFree(dCamViewAngleCos);
    cudaFree(dObjInJudgeAry);
}

__global__ void glpaGpuRender(
    float* polyVs,
    float* polyNs,
    int polyAmount,
    float* mtCamTransRot,
    float* mtCamRot,
    float camFarZ,
    float camNearZ,
    float* camViewAngleCos,
    float* viewVolumeVs,
    float* viewVolumeNs,
    float* nearScSize,
    float* scPixelSize,
    float* result,
    float* debugFloatAry
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyAmount)
    {
        float vec3d[3] = {0};

        vec3d[AX] = polyVs[i*9 + AX];
        vec3d[AY] = polyVs[i*9 + AY];
        vec3d[AZ] = polyVs[i*9 + AZ];
        float cnvtPolyV1[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV1);

        vec3d[AX] = polyVs[i*9 + 3 + AX];
        vec3d[AY] = polyVs[i*9 + 3 + AY];
        vec3d[AZ] = polyVs[i*9 + 3 + AZ];
        float cnvtPolyV2[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV2);

        vec3d[AX] = polyVs[i*9 + 6 + AX];
        vec3d[AY] = polyVs[i*9 + 6 + AY];
        vec3d[AZ] = polyVs[i*9 + 6 + AZ];
        float cnvtPolyV3[3];
        MT_PRODUCT_4X4_VEC3D(mtCamTransRot, vec3d, cnvtPolyV3);

        float cnvtPolyVs[9] = {
            cnvtPolyV1[AX], cnvtPolyV1[AY], cnvtPolyV1[AZ],
            cnvtPolyV2[AX], cnvtPolyV2[AY], cnvtPolyV2[AZ],
            cnvtPolyV3[AX], cnvtPolyV3[AY], cnvtPolyV3[AZ]
        };

        vec3d[AX] = polyNs[i*3 + AX];
        vec3d[AY] = polyNs[i*3 + AY];
        vec3d[AZ] = polyNs[i*3 + AZ];
        float cnvtPolyN[3];
        MT_PRODUCT_4X4_VEC3D(mtCamRot, vec3d, cnvtPolyN);

        float polyVxPolyNDotCos;
        VEC_GET_VECS_COS(cnvtPolyV1, cnvtPolyN, polyVxPolyNDotCos);
        
        int polyBilateralIF = (polyVxPolyNDotCos <= 0) ? TRUE : FALSE;

        for (int conditionalBranch = 0; conditionalBranch < polyBilateralIF; conditionalBranch++)
        {
            int assignAryNum = 0;
            float inxtn[MAX_VIEW_VOLUE_POLY_INXTN*3] = {0};
            float pixelInxtn[MAX_VIEW_VOLUE_POLY_INXTN*2] = {-1};
            
            int polyV1InIF = 0;
            int polyV2InIF = 0;
            int polyV3InIF = 0;
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV1, camFarZ, camNearZ, camViewAngleCos, polyV1InIF);
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV2, camFarZ, camNearZ, camViewAngleCos, polyV2InIF);
            JUDGE_POLY_V_IN_VIEW_VOLUME(cnvtPolyV3, camFarZ, camNearZ, camViewAngleCos, polyV3InIF);

            for (int conditionalBranch2 = 0; conditionalBranch2 < polyV1InIF; conditionalBranch2++)
            {
                inxtn[assignAryNum*3 + AX] = cnvtPolyV1[AX];
                inxtn[assignAryNum*3 + AY] = cnvtPolyV1[AY];
                inxtn[assignAryNum*3 + AZ] = cnvtPolyV1[AZ];
                VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], cnvtPolyV1, 0, camNearZ, nearScSize, scPixelSize);
                VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], cnvtPolyV1, 0, camNearZ, nearScSize, scPixelSize);
                assignAryNum++;
            }

            for (int conditionalBranch2 = 0; conditionalBranch2 < polyV2InIF; conditionalBranch2++)
            {
                inxtn[assignAryNum*3 + AX] = cnvtPolyV2[AX];
                inxtn[assignAryNum*3 + AY] = cnvtPolyV2[AY];
                inxtn[assignAryNum*3 + AZ] = cnvtPolyV2[AZ];
                VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], cnvtPolyV2, 0, camNearZ, nearScSize, scPixelSize);
                VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], cnvtPolyV2, 0, camNearZ, nearScSize, scPixelSize);
                assignAryNum++;
            }

            for (int conditionalBranch2 = 0; conditionalBranch2 < polyV3InIF; conditionalBranch2++)
            {
                inxtn[assignAryNum*3 + AX] = cnvtPolyV3[AX];
                inxtn[assignAryNum*3 + AY] = cnvtPolyV3[AY];
                inxtn[assignAryNum*3 + AZ] = cnvtPolyV3[AZ];
                VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], cnvtPolyV3, 0, camNearZ, nearScSize, scPixelSize);
                VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], cnvtPolyV3, 0, camNearZ, nearScSize, scPixelSize);
                assignAryNum++;
            }

            int noVsInIF = (polyV1InIF == FALSE && polyV2InIF == FALSE && polyV3InIF == FALSE) ? TRUE : FALSE;
            int shapeCnvtIF = ((polyV1InIF + polyV2InIF + polyV3InIF != 3) && noVsInIF == FALSE) ? TRUE : FALSE;
            for (int conditionalBranch2 = 0; conditionalBranch2 < noVsInIF; conditionalBranch2++)
            {
                float polyRectOrigin[3] = {cnvtPolyV1[AX], cnvtPolyV1[AY], cnvtPolyV1[AZ]};
                float polyRectOpposite[3] = {cnvtPolyV1[AX], cnvtPolyV1[AY], cnvtPolyV1[AZ]};

                polyRectOrigin[AX] = (cnvtPolyV2[AX] < polyRectOrigin[AX]) ? cnvtPolyV2[AX] : polyRectOrigin[AX];
                polyRectOrigin[AY] = (cnvtPolyV2[AY] < polyRectOrigin[AY]) ? cnvtPolyV2[AY] : polyRectOrigin[AY];
                polyRectOrigin[AZ] = (cnvtPolyV2[AZ] > polyRectOrigin[AZ]) ? cnvtPolyV2[AZ] : polyRectOrigin[AZ];

                polyRectOpposite[AX] = (cnvtPolyV2[AX] > polyRectOpposite[AX]) ? cnvtPolyV2[AX] : polyRectOpposite[AX];
                polyRectOpposite[AY] = (cnvtPolyV2[AY] > polyRectOpposite[AY]) ? cnvtPolyV2[AY] : polyRectOpposite[AY];
                polyRectOpposite[AZ] = (cnvtPolyV2[AZ] < polyRectOpposite[AZ]) ? cnvtPolyV2[AZ] : polyRectOpposite[AZ];

                polyRectOrigin[AX] = (cnvtPolyV3[AX] < polyRectOrigin[AX]) ? cnvtPolyV3[AX] : polyRectOrigin[AX];
                polyRectOrigin[AY] = (cnvtPolyV3[AY] < polyRectOrigin[AY]) ? cnvtPolyV3[AY] : polyRectOrigin[AY];
                polyRectOrigin[AZ] = (cnvtPolyV3[AZ] > polyRectOrigin[AZ]) ? cnvtPolyV3[AZ] : polyRectOrigin[AZ];

                polyRectOpposite[AX] = (cnvtPolyV3[AX] > polyRectOpposite[AX]) ? cnvtPolyV3[AX] : polyRectOpposite[AX];
                polyRectOpposite[AY] = (cnvtPolyV3[AY] > polyRectOpposite[AY]) ? cnvtPolyV3[AY] : polyRectOpposite[AY];
                polyRectOpposite[AZ] = (cnvtPolyV3[AZ] < polyRectOpposite[AZ]) ? cnvtPolyV3[AZ] : polyRectOpposite[AZ];

                // TODO: 3 and 4 are different from the source. This may be the cause of the bug, so please check.
                float polyRectOppositeSideVs[12] = {
                    polyRectOrigin[AX], 0,  polyRectOpposite[AZ],
                    polyRectOpposite[AX], 0, polyRectOpposite[AZ],
                    0, polyRectOrigin[AY], polyRectOpposite[AZ],
                    0, polyRectOpposite[AY], polyRectOpposite[AZ]
                };

                float zVec[3] = {0, 0, -1};
                float vecsCos[4];

                for (int aryI = 0; aryI < 4; aryI++)
                {
                    float calcObjOppositeV[3] = {
                        polyRectOppositeSideVs[aryI*3 + AX],
                        polyRectOppositeSideVs[aryI*3 + AY],
                        polyRectOppositeSideVs[aryI*3 + AZ]
                    };

                    VEC_GET_VECS_COS(zVec, calcObjOppositeV, vecsCos[aryI]);
                }

                int polyZInIF = (polyRectOrigin[AZ] >= -camFarZ && polyRectOpposite[AZ] <= -camNearZ) ? TRUE : FALSE;

                // True if positive, false if negative.
                int xzOriginSymbol = (polyRectOrigin[AX] >= 0) ? TRUE : FALSE;
                int xzOppositeSymbol = (polyRectOrigin[AX] >= 0) ? TRUE : FALSE;
                int yzOriginSymbol = (polyRectOrigin[AY] >= 0) ? TRUE : FALSE;
                int yzOppositeSymbol = (polyRectOrigin[AY] >= 0) ? TRUE : FALSE;

                int polyXzInIF = 
                (
                    (xzOriginSymbol == TRUE && vecsCos[0] >= camViewAngleCos[AX]) || 
                    (xzOriginSymbol == FALSE && xzOppositeSymbol == TRUE) ||
                    (xzOppositeSymbol == TRUE && vecsCos[1] >= camViewAngleCos[AX])
                ) ? TRUE : FALSE;

                int polyYzInIF = 
                (
                    (yzOriginSymbol == TRUE && vecsCos[2] >= camViewAngleCos[AY]) || 
                    (yzOriginSymbol == FALSE && yzOppositeSymbol == TRUE) || 
                    (xzOppositeSymbol == TRUE && vecsCos[3] >= camViewAngleCos[AY])
                ) ? TRUE : FALSE;

                shapeCnvtIF = (polyZInIF == TRUE && polyXzInIF == TRUE && polyYzInIF == TRUE) ? TRUE : FALSE;
            }

            for(int conditionalBranch2 = 0; conditionalBranch2 < shapeCnvtIF; conditionalBranch2++)
            {
                int vvFaceI[6] = {
                    RECT_FRONT_TOP_LEFT,
                    RECT_FRONT_TOP_LEFT,
                    RECT_BACK_BOTTOM_RIGHT,
                    RECT_FRONT_TOP_LEFT,
                    RECT_BACK_BOTTOM_RIGHT,
                    RECT_BACK_BOTTOM_RIGHT
                };

                int vvFaceVsI[24] = {
                    VIEWVOLUME_TOP_V1, VIEWVOLUME_TOP_V2, VIEWVOLUME_TOP_V3, VIEWVOLUME_TOP_V4,
                    VIEWVOLUME_FRONT_V1, VIEWVOLUME_FRONT_V2, VIEWVOLUME_FRONT_V3, VIEWVOLUME_FRONT_V4,
                    VIEWVOLUME_RIGHT_V1, VIEWVOLUME_RIGHT_V2, VIEWVOLUME_RIGHT_V3, VIEWVOLUME_RIGHT_V4,
                    VIEWVOLUME_LEFT_V1, VIEWVOLUME_LEFT_V2, VIEWVOLUME_LEFT_V3, VIEWVOLUME_LEFT_V4,
                    VIEWVOLUME_BACK_V1, VIEWVOLUME_BACK_V2, VIEWVOLUME_BACK_V3, VIEWVOLUME_BACK_V4,
                    VIEWVOLUME_BOTTOM_V1, VIEWVOLUME_BOTTOM_V2, VIEWVOLUME_BOTTOM_V3, VIEWVOLUME_BOTTOM_V4
                };

                int vvLineVI[24] = {
                    RECT_L1_STARTV, RECT_L1_ENDV,
                    RECT_L2_STARTV, RECT_L2_ENDV,
                    RECT_L3_STARTV, RECT_L3_ENDV,
                    RECT_L4_STARTV, RECT_L4_ENDV,
                    RECT_L5_STARTV, RECT_L5_ENDV,
                    RECT_L6_STARTV, RECT_L6_ENDV,
                    RECT_L7_STARTV, RECT_L7_ENDV,
                    RECT_L8_STARTV, RECT_L8_ENDV,
                    RECT_L9_STARTV, RECT_L9_ENDV,
                    RECT_L10_STARTV, RECT_L10_ENDV,
                    RECT_L11_STARTV, RECT_L11_ENDV,
                    RECT_L12_STARTV, RECT_L12_ENDV
                };

                // Obtain the intersection between the view volume line and the polygon surface.
                for (int roopLineI = 0; roopLineI < 12; roopLineI++)
                {
                    float polyFaceDot[2];
                    polyFaceDot[0] = 
                        (viewVolumeVs[vvLineVI[roopLineI*2]*3 + AX] - cnvtPolyV1[AX]) * cnvtPolyN[AX] + 
                        (viewVolumeVs[vvLineVI[roopLineI*2]*3 + AY] - cnvtPolyV1[AY]) * cnvtPolyN[AY] + 
                        (viewVolumeVs[vvLineVI[roopLineI*2]*3 + AZ] - cnvtPolyV1[AZ]) * cnvtPolyN[AZ];
                    polyFaceDot[1] = 
                        (viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AX] - cnvtPolyV1[AX]) * cnvtPolyN[AX] + 
                        (viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AY] - cnvtPolyV1[AY]) * cnvtPolyN[AY] + 
                        (viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AZ] - cnvtPolyV1[AZ]) * cnvtPolyN[AZ];

                    int vAllOnFaceIF = (polyFaceDot[0] == 0 && polyFaceDot[1] == 0) ? TRUE : FALSE;
                    int v1OnFaceIF = (polyFaceDot[0] == 0 && vAllOnFaceIF == FALSE) ? TRUE : FALSE;
                    int v2OnFaceIF = (polyFaceDot[1] == 0 && vAllOnFaceIF == FALSE) ? TRUE : FALSE;

                    for (int conditionalBranch3 = 0; conditionalBranch3 < vAllOnFaceIF; conditionalBranch3++)
                    {
                        inxtn[assignAryNum*3 + AX] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AX];
                        inxtn[assignAryNum*3 + AY] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AY];
                        inxtn[assignAryNum*3 + AZ] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AZ];
                        VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], viewVolumeVs, vvLineVI[roopLineI*2]*3, camNearZ, nearScSize, scPixelSize);
                        VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], viewVolumeVs, vvLineVI[roopLineI*2]*3, camNearZ, nearScSize, scPixelSize);
                        assignAryNum++;

                        inxtn[assignAryNum*3 + AX] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AX];
                        inxtn[assignAryNum*3 + AY] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AY];
                        inxtn[assignAryNum*3 + AZ] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AZ];
                        VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], viewVolumeVs, vvLineVI[roopLineI*2 + 1], camNearZ, nearScSize, scPixelSize);
                        VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], viewVolumeVs, vvLineVI[roopLineI*2 + 1], camNearZ, nearScSize, scPixelSize);
                        assignAryNum++;
                    }

                    for (int conditionalBranch3 = 0; conditionalBranch3 < v1OnFaceIF; conditionalBranch3++)
                    {
                        inxtn[assignAryNum*3 + AX] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AX];
                        inxtn[assignAryNum*3 + AY] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AY];
                        inxtn[assignAryNum*3 + AZ] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + AZ];
                        VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], viewVolumeVs, vvLineVI[roopLineI*2]*3, camNearZ, nearScSize, scPixelSize);
                        VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], viewVolumeVs, vvLineVI[roopLineI*2]*3, camNearZ, nearScSize, scPixelSize);
                        assignAryNum++;
                    }

                    for (int conditionalBranch3 = 0; conditionalBranch3 < v2OnFaceIF; conditionalBranch3++)
                    {
                        inxtn[assignAryNum*3 + AX] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AX];
                        inxtn[assignAryNum*3 + AY] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AY];
                        inxtn[assignAryNum*3 + AZ] = viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + AZ];
                        VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], viewVolumeVs, vvLineVI[roopLineI*2 + 1], camNearZ, nearScSize, scPixelSize);
                        VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], viewVolumeVs, vvLineVI[roopLineI*2 + 1], camNearZ, nearScSize, scPixelSize);
                        assignAryNum++;
                    }
                    
                    int calcInxtnIF = ((polyFaceDot[0] > 0 && polyFaceDot[1] < 0) || (polyFaceDot[0] < 0 && polyFaceDot[1] > 0)) ? TRUE : FALSE; 

                    for(int conditionalBranch3 = 0; conditionalBranch3 < calcInxtnIF; conditionalBranch3++) 
                    { 
                        float calcInxtn[3];
                        for (int roopCoord = 0; roopCoord < 3; roopCoord++) 
                        { 
                            calcInxtn[roopCoord] = viewVolumeVs[vvLineVI[roopLineI*2]*3 + roopCoord] + 
                                (viewVolumeVs[vvLineVI[roopLineI*2 + 1]*3 + roopCoord] - viewVolumeVs[vvLineVI[roopLineI*2]*3 + roopCoord]) * 
                                (fabs(polyFaceDot[0]) / (fabs(polyFaceDot[0]) + fabs(polyFaceDot[1])));
                        } 
                        
                        float vecCos[6];
                        CALC_VEC_COS(vecCos[0], cnvtPolyV1, cnvtPolyV2, cnvtPolyV1, calcInxtn);
                        CALC_VEC_COS(vecCos[1], cnvtPolyV1, cnvtPolyV2, cnvtPolyV1, cnvtPolyV3);
                        CALC_VEC_COS(vecCos[2], cnvtPolyV2, cnvtPolyV3, cnvtPolyV2, calcInxtn);
                        CALC_VEC_COS(vecCos[3], cnvtPolyV2, cnvtPolyV3, cnvtPolyV2, cnvtPolyV1);
                        CALC_VEC_COS(vecCos[4], cnvtPolyV3, cnvtPolyV1, cnvtPolyV3, calcInxtn);
                        CALC_VEC_COS(vecCos[5], cnvtPolyV3, cnvtPolyV1, cnvtPolyV3, cnvtPolyV2);
                        
                        int inxtnInPolyFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5]) ? TRUE : FALSE;
                        
                        for (int conditionalBranch4 = 0; conditionalBranch4 < inxtnInPolyFaceIF; conditionalBranch4++) 
                        { 
                            inxtn[assignAryNum*3 + AX] = calcInxtn[AX];
                            inxtn[assignAryNum*3 + AY] = calcInxtn[AY];
                            inxtn[assignAryNum*3 + AZ] = calcInxtn[AZ];
                            VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], calcInxtn, 0, camNearZ, nearScSize, scPixelSize);
                            VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], calcInxtn, 0, camNearZ, nearScSize, scPixelSize);

                            // debugFloatAry[i*roopLineI*6 + roopLineI*6 + AX] = inxtn[assignAryNum*3 + AX];
                            // debugFloatAry[i*roopLineI*6 + roopLineI*6 + AY] = inxtn[assignAryNum*3 + AY];
                            // debugFloatAry[i*roopLineI*6 + roopLineI*6 + AZ] = inxtn[assignAryNum*3 + AZ];
                            // debugFloatAry[i*roopLineI*6 + roopLineI*6 + 3] = pixelInxtn[assignAryNum*2 + AX];
                            // debugFloatAry[i*roopLineI*6 + roopLineI*6 + 4] = pixelInxtn[assignAryNum*2 + AY];

                            assignAryNum++;
                        } 
                    } 
                    // debugFloatAry[i*roopLineI*6 + roopLineI*6 + 5] = (i >= 12) ? i - 12 : i;

                    
                }

                for (int roopFaceI = 0; roopFaceI < 6; roopFaceI++)
                {
                    for (int startPolyVI = 0; startPolyVI < 3; startPolyVI++)
                    {
                        // int debugCosNotCalc = TRUE;

                        int endPolyVI = (startPolyVI == 2) ? 0 : startPolyVI + 1;

                        float startPolyV[3] ={
                            cnvtPolyVs[startPolyVI*3 + AX], cnvtPolyVs[startPolyVI*3 + AY], cnvtPolyVs[startPolyVI*3 + AZ]
                        };

                        float endPolyV[3] ={
                            cnvtPolyVs[endPolyVI*3 + AX], cnvtPolyVs[endPolyVI*3 + AY], cnvtPolyVs[endPolyVI*3 + AZ]
                        };

                        float vvFaceDot[2];
                        vvFaceDot[0] = 
                            (startPolyV[AX] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AX]) * viewVolumeNs[roopFaceI*3 + AX] + 
                            (startPolyV[AY] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AY]) * viewVolumeNs[roopFaceI*3 + AY] + 
                            (startPolyV[AZ] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AZ]) * viewVolumeNs[roopFaceI*3 + AZ];

                        vvFaceDot[1] = 
                            (endPolyV[AX] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AX]) * viewVolumeNs[roopFaceI*3 + AX] + 
                            (endPolyV[AY] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AY]) * viewVolumeNs[roopFaceI*3 + AY] + 
                            (endPolyV[AZ] - viewVolumeVs[vvFaceI[roopFaceI]*3 + AZ]) * viewVolumeNs[roopFaceI*3 + AZ];

                        int vAllOnFaceIF = (vvFaceDot[0] == 0 && vvFaceDot[1] == 0) ? TRUE : FALSE;
                        int startPolyVOnFaceIF = (vvFaceDot[0] == 0 && vAllOnFaceIF == FALSE) ? TRUE : FALSE;
                        int endPolyVOnFaceIF = (vvFaceDot[1] == 0 && vAllOnFaceIF == FALSE) ? TRUE : FALSE;

                        for (int conditionalBranch3 = 0; conditionalBranch3 < vAllOnFaceIF; conditionalBranch3++)
                        {
                            inxtn[assignAryNum*3 + AX] = startPolyV[AX];
                            inxtn[assignAryNum*3 + AY] = startPolyV[AY];
                            inxtn[assignAryNum*3 + AZ] = startPolyV[AZ];
                            VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], startPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], startPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            assignAryNum++;

                            inxtn[assignAryNum*3 + AX] = endPolyV[AX];
                            inxtn[assignAryNum*3 + AY] = endPolyV[AY];
                            inxtn[assignAryNum*3 + AZ] = endPolyV[AZ];
                            VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], endPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], endPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            assignAryNum++;
                        }

                        for (int conditionalBranch3 = 0; conditionalBranch3 < startPolyVOnFaceIF; conditionalBranch3++)
                        {
                            inxtn[assignAryNum*3 + AX] = startPolyV[AX];
                            inxtn[assignAryNum*3 + AY] = startPolyV[AY];
                            inxtn[assignAryNum*3 + AZ] = startPolyV[AZ];
                            VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], startPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], startPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            assignAryNum++;
                        }

                        for (int conditionalBranch3 = 0; conditionalBranch3 < endPolyVOnFaceIF; conditionalBranch3++)
                        {
                            inxtn[assignAryNum*3 + AX] = endPolyV[AX];
                            inxtn[assignAryNum*3 + AY] = endPolyV[AY];
                            inxtn[assignAryNum*3 + AZ] = endPolyV[AZ];
                            VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], endPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], endPolyV, 0, camNearZ, nearScSize, scPixelSize);
                            assignAryNum++;
                        }

                        int calcInxtnIF  = ((vvFaceDot[0] > 0 && vvFaceDot[1] < 0) || (vvFaceDot[0] < 0 && vvFaceDot[1] > 0)) ? TRUE : FALSE;
                        // debugFloatAry[i*54 + roopFaceI*9 + startPolyVI*3] = 0;
                        // debugFloatAry[i*54 + roopFaceI*9 + startPolyVI*3 + 1] = calcInxtnIF;
                        // debugFloatAry[i*54 + roopFaceI*9 + startPolyVI*3 + 2] = (i >= 12) ? i - 12 : i;

                        int inxtnInVvFaceIF = 0;
                        for(int conditionalBranch3 = 0; conditionalBranch3 < calcInxtnIF; conditionalBranch3++) 
                        { 
                            float calcInxtn[3];
                            for (int roopCoord = 0; roopCoord < 3; roopCoord++) 
                            { 
                                calcInxtn[roopCoord] = startPolyV[roopCoord] + 
                                    (endPolyV[roopCoord] - startPolyV[roopCoord]) * 
                                    (fabs(vvFaceDot[0]) / (fabs(vvFaceDot[0]) + fabs(vvFaceDot[1])));
                            } 
                            
                            float vecCos[8];
                            CALC_VEC_ARY_COS(vecCos[0], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, calcInxtn, 0);
                            CALC_VEC_ARY_COS(vecCos[1], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3);
                            
                            CALC_VEC_ARY_COS(vecCos[2], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3, calcInxtn, 0);
                            CALC_VEC_ARY_COS(vecCos[3], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3);
                            
                            CALC_VEC_ARY_COS(vecCos[4], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, calcInxtn, 0);
                            CALC_VEC_ARY_COS(vecCos[5], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V2]*3);
                            
                            CALC_VEC_ARY_COS(vecCos[6], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, calcInxtn, 0);
                            CALC_VEC_ARY_COS(vecCos[7], viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V1]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V4]*3, viewVolumeVs, vvFaceVsI[roopFaceI*4 + FACE_V3]*3);

                            // debugCosNotCalc = FALSE;
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9] = vecCos[0];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 1] = vecCos[1];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 2] = vecCos[2];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 3] = vecCos[3];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 4] = vecCos[4];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 5] = vecCos[5];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 6] = vecCos[6];
                            // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 7] = vecCos[7];

                            
                            inxtnInVvFaceIF = (vecCos[0] >= vecCos[1] && vecCos[2] >= vecCos[3] && vecCos[4] >= vecCos[5] && vecCos[6] >= vecCos[7]) ? TRUE : FALSE;
                            
                            for (int conditionalBranch4 = 0; conditionalBranch4 < inxtnInVvFaceIF; conditionalBranch4++) 
                            { 
                                inxtn[assignAryNum*3 + AX] = calcInxtn[AX];
                                inxtn[assignAryNum*3 + AY] = calcInxtn[AY];
                                inxtn[assignAryNum*3 + AZ] = calcInxtn[AZ];
                                VX_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AX], calcInxtn, 0, camNearZ, nearScSize, scPixelSize);
                                VY_SCREEN_PIXEL_CONVERT(pixelInxtn[assignAryNum*2 + AY], calcInxtn, 0, camNearZ, nearScSize, scPixelSize);

                                // debugCosNotCalc = FALSE;
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9] = 1000;
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 1] = inxtn[assignAryNum*3 + AX];
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 2] = inxtn[assignAryNum*3 + AY];
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 3] = inxtn[assignAryNum*3 + AZ];
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 4] = (pixelInxtn[assignAryNum*2 + AX] == 0) ? 0.001 : pixelInxtn[assignAryNum*2 + AX];
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 5] = (pixelInxtn[assignAryNum*2 + AY] == 0) ? 0.001 : pixelInxtn[assignAryNum*2 + AY];
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 6] = startPolyVI + 1;
                                // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 7] = roopFaceI + 1;

                                assignAryNum++;

                            } 
                        } 

                        // for (int conditionalBranch4 = 0; conditionalBranch4 < debugCosNotCalc; conditionalBranch4++)
                        // {
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 1] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 2] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 3] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 4] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 5] = -1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 6] = startPolyVI + 1;
                        //     debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 7] = roopFaceI + 1;
                        // }
                        // debugFloatAry[i*162 + roopFaceI*27 + startPolyVI*9 + 8] = (i >= 12) ? i - 12 : i;
                    }
                }
            }
            
            int polyInIF = ((noVsInIF == FALSE || shapeCnvtIF == TRUE) && pixelInxtn[FACE_V3*2 + 1] != -1) ? TRUE : FALSE;

            for (int conditionalBranch2 = 0; conditionalBranch2 < polyInIF; conditionalBranch2++)
            {
                int inxtnAmount = 3;
                for (int vAryI = FACE_V4; vAryI < MAX_VIEW_VOLUE_POLY_INXTN; vAryI++)
                {
                    
                }
            }

        }
    }
}

void Render::rasterize(std::unordered_map<std::wstring, Object> sObj, Camera cam, LPDWORD buffer)
{
    std::vector<float> polyVs;
    std::vector<float> polyNs;
    int loopObjI = 0;
    for (auto obj : sObj)
    {
        loopObjI++;
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

    float* dPolyVs;
    float* dPolyNs;
    cudaMalloc((void**)&dPolyVs, sizeof(float)*polyVs.size());
    cudaMalloc((void**)&dPolyNs, sizeof(float)*polyNs.size());
    cudaMemcpy(dPolyVs, polyVs.data(), sizeof(float)*polyVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyNs, polyNs.data(), sizeof(float)*polyNs.size(), cudaMemcpyHostToDevice);


    float* dMtCamTransRot;
    cudaMalloc((void**)&dMtCamTransRot, sizeof(float)*hMtCamTransRot.size());
    cudaMemcpy(dMtCamTransRot, hMtCamTransRot.data(), sizeof(float)*hMtCamTransRot.size(), cudaMemcpyHostToDevice);


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

    float* dMtCamRot;
    cudaMalloc((void**)&dMtCamRot, sizeof(float)*hMtCamRot.size());
    cudaMemcpy(dMtCamRot, hMtCamRot.data(), sizeof(float)*hMtCamRot.size(), cudaMemcpyHostToDevice);


    float* dCamViewAngleCos;
    cudaMalloc((void**)&dCamViewAngleCos, sizeof(float)*hCamViewAngleCos.size());
    cudaMemcpy(dCamViewAngleCos, hCamViewAngleCos.data(), sizeof(float)*hCamViewAngleCos.size(), cudaMemcpyHostToDevice);


    std::vector<float> hViewVolumeVs;
    for (int i = 0; i < 8; i++){
        hViewVolumeVs.push_back(cam.viewVolume.v[i].x / CALC_SCALE);
        hViewVolumeVs.push_back(cam.viewVolume.v[i].y / CALC_SCALE);
        hViewVolumeVs.push_back(cam.viewVolume.v[i].z / CALC_SCALE);
    }

    float* dViewVolumeVs;
    cudaMalloc((void**)&dViewVolumeVs, sizeof(float)*hViewVolumeVs.size());
    cudaMemcpy(dViewVolumeVs, hViewVolumeVs.data(), sizeof(float)*hViewVolumeVs.size(), cudaMemcpyHostToDevice);


    std::vector<float> hViewVolumeNs;
    for (int i = 0; i < 6; i++){
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].x / CALC_SCALE);
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].y / CALC_SCALE);
        hViewVolumeNs.push_back(cam.viewVolume.face.normal[i].z / CALC_SCALE);
    }

    float* dViewVolumeNs;
    cudaMalloc((void**)&dViewVolumeNs, sizeof(float)*hViewVolumeNs.size());
    cudaMemcpy(dViewVolumeNs, hViewVolumeNs.data(), sizeof(float)*hViewVolumeNs.size(), cudaMemcpyHostToDevice);


    std::vector<float> hNearScSize;
    hNearScSize.push_back(cam.nearScrSize.x / CALC_SCALE);
    hNearScSize.push_back(cam.nearScrSize.y / CALC_SCALE);

    float* dNearScSize;
    cudaMalloc((void**)&dNearScSize, sizeof(float)*hNearScSize.size());
    cudaMemcpy(dNearScSize, hNearScSize.data(), sizeof(float)*hNearScSize.size(), cudaMemcpyHostToDevice);

    
    std::vector<float> hScPixelSize;
    hScPixelSize.push_back(cam.scPixelSize.x);
    hScPixelSize.push_back(cam.scPixelSize.y);

    float* dScPixelSize;
    cudaMalloc((void**)&dScPixelSize, sizeof(float)*hScPixelSize.size());
    cudaMemcpy(dScPixelSize, hScPixelSize.data(), sizeof(float)*hScPixelSize.size(), cudaMemcpyHostToDevice);

    int resultSize = (polyNs.size() / 3) * (12 * 3 * 3 + 3*3 + 3*3);
    float* hResult = new float[resultSize];
    float* dResult;
    cudaMalloc((void**)&dResult, sizeof(float)*resultSize);

    int polyAmount = polyNs.size() / 3;

    float camFarZ = static_cast<float>(cam.farZ / CALC_SCALE);
    float camNearZ = static_cast<float>(cam.nearZ / CALC_SCALE);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int dataSize = polyNs.size() / 3;
    int desiredThreadsPerBlock = 256;

    int blocks = (dataSize + desiredThreadsPerBlock - 1) / desiredThreadsPerBlock;
    int threadsPerBlock = std::min(desiredThreadsPerBlock, deviceProp.maxThreadsPerBlock);  

    dim3 dimBlock(threadsPerBlock);
    dim3 dimGrid(blocks);

    // int debugArySize = polyAmount * 9 * 6;
    int debugArySize = polyAmount * 162;
    // int debugArySize = polyAmount * 12 * 6;
    float* hDebugAry = new float[debugArySize];
    float* dDebugAry;
    cudaMalloc((void**)&dDebugAry, sizeof(float)*debugArySize);

    glpaGpuRender<<<dimGrid, dimBlock>>>
    (
        dPolyVs, dPolyNs, polyAmount, 
        dMtCamTransRot, dMtCamRot, camFarZ, camNearZ,
        dCamViewAngleCos, dViewVolumeVs, dViewVolumeNs, dNearScSize, dScPixelSize, dResult, dDebugAry
    );
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != 0)
    {
        throw std::runtime_error(ERROR_VECTOR_CUDA_ERROR);
    }
    
    cudaMemcpy(hDebugAry, dDebugAry, sizeof(float)*debugArySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hResult, dResult, sizeof(float)*resultSize , cudaMemcpyDeviceToHost);

    delete[] hObjInJudgeAry;
    delete[] hResult;

    cudaFree(dPolyVs);
    cudaFree(dPolyNs);
    cudaFree(dMtCamTransRot);
    cudaFree(dMtCamRot);
    cudaFree(dCamViewAngleCos);
    cudaFree(dViewVolumeVs);
    cudaFree(dViewVolumeNs);
    cudaFree(dNearScSize);
    cudaFree(dScPixelSize);
    cudaFree(dResult);


}
