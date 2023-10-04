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
        viewPointXZ.resize(4);
        viewPointYZ.resize(4);

        initialized = true;
    }

    // define screen size
    nearScreenSize.width = tan(viewAngle / 2 * PI / 180) * nearZ * 2;
    nearScreenSize.height = nearScreenSize.width * aspectRatio.y / aspectRatio.x;

    farScreenSize.width = nearScreenSize.width / 2 * farZ / nearZ;
    farScreenSize.height = farScreenSize.width * aspectRatio.y / aspectRatio.x;
    
    // Define coordinates of view area vertices on xz axis
    viewPointXZ[VP1].x = -nearScreenSize.width / 2;
    viewPointXZ[VP2].x = -farScreenSize.width / 2;
    viewPointXZ[VP3].x = farScreenSize.width / 2;
    viewPointXZ[VP4].x = nearScreenSize.width / 2;

    viewPointXZ[VP1].z = nearZ;
    viewPointXZ[VP2].z = farZ;
    viewPointXZ[VP3].z = farZ;
    viewPointXZ[VP4].z = nearZ;

    // Define coordinates of view area vertices on yz axis
    viewPointYZ[VP1].y = nearScreenSize.height / 2;
    viewPointYZ[VP2].y = farScreenSize.height / 2;
    viewPointYZ[VP3].y = -farScreenSize.height / 2;
    viewPointYZ[VP4].y = -nearScreenSize.height / 2;

    viewPointYZ[VP1].z = nearZ;
    viewPointYZ[VP2].z = farZ;
    viewPointYZ[VP3].z = farZ;
    viewPointYZ[VP4].z = nearZ;

    #ifdef DEBUG_CAMERA_

    std::vector<VECTOR3D> transViewPoint;
    std::vector<VECTOR3D> transedViewPoint;
    transViewPoint.resize(8);
    // 0
    transViewPoint[0].x = viewPointXZ[VP1].x;
    transViewPoint[0].y = viewPointYZ[VP1].y;
    transViewPoint[0].z = nearZ;

    // 1
    transViewPoint[1].x = viewPointXZ[VP4].x;
    transViewPoint[1].y = viewPointYZ[VP1].y;
    transViewPoint[1].z = nearZ;

    // 2
    transViewPoint[2].x = viewPointXZ[VP4].x;
    transViewPoint[2].y = viewPointYZ[VP4].y;
    transViewPoint[2].z = nearZ;

    // 3
    transViewPoint[3].x = viewPointXZ[VP1].x;
    transViewPoint[3].y = viewPointYZ[VP4].y;
    transViewPoint[3].z = nearZ;

    // 4
    transViewPoint[4].x = viewPointXZ[VP2].x;
    transViewPoint[4].y = viewPointYZ[VP2].y;
    transViewPoint[4].z = farZ;

    // 5
    transViewPoint[5].x = viewPointXZ[VP3].x;
    transViewPoint[5].y = viewPointYZ[VP2].y;
    transViewPoint[5].z = farZ;

    // 6
    transViewPoint[6].x = viewPointXZ[VP3].x;
    transViewPoint[6].y = viewPointYZ[VP3].y;
    transViewPoint[6].z = farZ;

    // 7
    transViewPoint[7].x = viewPointXZ[VP2].x;
    transViewPoint[7].y = viewPointYZ[VP3].y;
    transViewPoint[7].z = farZ;

    mtx.posTrans(transViewPoint, wPos);
    transedViewPoint = mtx.resultMatrices;
    mtx.rotTrans(transedViewPoint, rotAngle);
    transedViewPoint = mtx.resultMatrices;

    VECTOR3D scaleRate = {2, 2, 2};
    mtx.scaleTrans(transedViewPoint, scaleRate);
    transedViewPoint = mtx.resultMatrices;

    #endif

}

void CAMERA::coordinateTransRange(std::vector<OBJ_FILE>& objData)
{
    for (int i = 0; i < objData.size(); ++i)
    {
        // world trans
        mtx.posTrans(objData[i].v.world, wPos);
        objData[i].calcV.world = mtx.resultMatrices;

        mtx.rotTrans(objData[i].v.world, rotAngle);


        // normal trans
        
    }
}

void CAMERA::clippingRange()
{

}

void CAMERA::polyBilateralJudge()
{

}

void CAMERA::coordinateTransV()
{
    
}

CAMERA mainCam;