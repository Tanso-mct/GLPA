#define DEBUG_CAMERA_
#include "camera.h"

void CAMERA::initialize()
{
    wPos = {0, 0, 0};
    rotAngle = {0, 0, 0};

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
    viewPointA[VP1].x = nearScreenSize.width / 2 * -1;
    viewPointA[VP2].x = nearScreenSize.width / 2;
    viewPointA[VP3].x = farScreenSize.width / 2;
    viewPointA[VP4].x = farScreenSize.width / 2 * -1;

    viewPointA[VP1].z = nearZ;
    viewPointA[VP2].z = nearZ;
    viewPointA[VP3].z = farZ;
    viewPointA[VP4].z = farZ;

    // Define coordinates of view area vertices on yz axis
    viewPointB[VP1].y = nearScreenSize.height / 2;
    viewPointB[VP2].y = nearScreenSize.height / 2 * -1;
    viewPointB[VP3].y = farScreenSize.height / 2 * -1;
    viewPointB[VP4].y = farScreenSize.height / 2;

    viewPointB[VP1].z = nearZ;
    viewPointB[VP2].z = nearZ;
    viewPointB[VP3].z = farZ;
    viewPointB[VP4].z = farZ;

    for (int i = 0; i < 4; ++i)
    {
        transViewPoint[i].x = viewPointA[i].x;
        transViewPoint[i].y = viewPointB[i].y;
        transViewPoint[i].z = viewPointB[i].z;
    }

    mtx.posTrans(transViewPoint, wPos);
    // transViewPoint = mtx.resultMatrices;
    // mtx.rotTrans(transViewPoint, SELECTAXIS_X, rotAngle.x);
    // transViewPoint = mtx.resultMatrices;
    // mtx.rotTrans(transViewPoint, SELECTAXIS_Y, rotAngle.y);
    // transViewPoint = mtx.resultMatrices;
    // mtx.rotTrans(transViewPoint, SELECTAXIS_Z, rotAngle.z);

    for (int i = 0; i < 4; ++i)
    {
        viewPointA[i].x = mtx.resultMatrices[i].x;
        viewPointB[i].y = mtx.resultMatrices[i].y;
        viewPointA[i].z = mtx.resultMatrices[i].z;
        viewPointB[i].z = mtx.resultMatrices[i].z;
    }

    #ifdef DEBUG_CAMERA_

    char buffer[100];

    OutputDebugStringA("view point a\n");
    for (int i = 0; i < 4; ++i)
    {
        snprintf(buffer, sizeof(buffer), "%.2lf", viewPointA[i].x);
        OutputDebugStringA("\n");
        OutputDebugStringA(buffer);
    }

    // OutputDebugStringA("\n");
    // OutputDebugStringA("view point b\n");
    // for (int i = 0; i < 4; ++i)
    // {
    //     sprintf_s(buffer, "%lf", viewPointB[i].y);
    //     OutputDebugStringA(buffer);
    //     OutputDebugStringA(" ");
    //     sprintf_s(buffer, "%lf", viewPointB[i].z);
    //     OutputDebugStringA(buffer);
    // }

    #endif

}

CAMERA mainCam;