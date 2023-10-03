#define DEBUG_CAMERA_
#include "camera.h"

void CAMERA::initialize()
{
    wPos = {0, 0, 0};
    rotAngle = {10, 0, 0};

    nearZ = 1;
    farZ = 1000;
    viewAngle = 80;
    aspectRatio = {16, 9};
}

void CAMERA::defClippingArea()
{
    // Processing to be done only for the first execution
    if (!initialized)
    {
        viewPointA.resize(4);
        viewPointB.resize(4);

        initialized = true;
    }

    // define screen size
    nearScreenSize.width = tan(viewAngle / 2 * PI / 180) * nearZ * 2;
    nearScreenSize.height = nearScreenSize.width * aspectRatio.y / aspectRatio.x;

    farScreenSize.width = nearScreenSize.width / 2 * farZ / nearZ;
    farScreenSize.height = farScreenSize.width * aspectRatio.y / aspectRatio.x;
    
    // Define coordinates of view area vertices on xz axis
    viewPointA[VP1].x = -nearScreenSize.width / 2;
    viewPointA[VP2].x = -farScreenSize.width / 2;
    viewPointA[VP3].x = farScreenSize.width / 2;
    viewPointA[VP4].x = nearScreenSize.width / 2;

    viewPointA[VP1].z = nearZ;
    viewPointA[VP2].z = farZ;
    viewPointA[VP3].z = farZ;
    viewPointA[VP4].z = nearZ;

    // Define coordinates of view area vertices on yz axis
    viewPointB[VP1].y = nearScreenSize.height / 2;
    viewPointB[VP2].y = farScreenSize.height / 2;
    viewPointB[VP3].y = -farScreenSize.height / 2;
    viewPointB[VP4].y = -nearScreenSize.height / 2;

    viewPointB[VP1].z = nearZ;
    viewPointB[VP2].z = farZ;
    viewPointB[VP3].z = farZ;
    viewPointB[VP4].z = nearZ;
    
    // 1
    transViewPoint[0].x = viewPointA[VP1].x;
    transViewPoint[0].y = viewPointB[VP1].y;
    transViewPoint[0].z = nearZ;

    // 2
    transViewPoint[1].x = viewPointA[VP4].x;
    transViewPoint[1].y = viewPointB[VP1].y;
    transViewPoint[1].z = nearZ;

    // 3
    transViewPoint[2].x = viewPointA[VP4].x;
    transViewPoint[2].y = viewPointB[VP4].y;
    transViewPoint[2].z = nearZ;

    // 4
    transViewPoint[3].x = viewPointA[VP1].x;
    transViewPoint[3].y = viewPointB[VP4].y;
    transViewPoint[3].z = nearZ;

    // 5
    transViewPoint[4].x = viewPointA[VP2].x;
    transViewPoint[4].y = viewPointB[VP2].y;
    transViewPoint[4].z = farZ;

    // 6
    transViewPoint[5].x = viewPointA[VP3].x;
    transViewPoint[5].y = viewPointB[VP2].y;
    transViewPoint[5].z = farZ;

    // 7
    transViewPoint[6].x = viewPointA[VP3].x;
    transViewPoint[6].y = viewPointB[VP3].y;
    transViewPoint[6].z = farZ;

    // 8
    transViewPoint[7].x = viewPointA[VP2].x;
    transViewPoint[7].y = viewPointB[VP3].y;
    transViewPoint[7].z = farZ;

    mtx.posTrans(transViewPoint, wPos);
    transViewPoint = mtx.resultMatrices;
    mtx.rotTrans(transViewPoint, SELECTAXIS_X, rotAngle.x);
    transViewPoint = mtx.resultMatrices;
    mtx.rotTrans(transViewPoint, SELECTAXIS_Y, rotAngle.y);
    transViewPoint = mtx.resultMatrices;
    mtx.rotTrans(transViewPoint, SELECTAXIS_Z, rotAngle.z);

    viewPointA[VP1].x = mtx.resultMatrices[0].x;
    viewPointA[VP1].z = mtx.resultMatrices[0].z;
    viewPointB[VP1].y = mtx.resultMatrices[0].y;
    viewPointB[VP1].z = mtx.resultMatrices[0].z;

    viewPointA[VP2].x = mtx.resultMatrices[4].x;
    viewPointA[VP2].z = mtx.resultMatrices[4].z;
    viewPointB[VP2].y = mtx.resultMatrices[4].y;
    viewPointB[VP2].z = mtx.resultMatrices[4].z;

    viewPointA[VP3].x = mtx.resultMatrices[5].x;
    viewPointA[VP3].z = mtx.resultMatrices[5].z;
    viewPointB[VP3].y = mtx.resultMatrices[6].y;
    viewPointB[VP3].z = mtx.resultMatrices[6].z;

    viewPointA[VP4].x = mtx.resultMatrices[1].x;
    viewPointA[VP4].z = mtx.resultMatrices[1].z;
    viewPointB[VP4].y = mtx.resultMatrices[2].y;
    viewPointB[VP4].z = mtx.resultMatrices[2].z;
}

CAMERA mainCam;