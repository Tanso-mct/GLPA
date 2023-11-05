#include "view_volume.h"

void ViewVolume::define
(
    double nearZ, double farZ,
    SIZE2* nearScrSize, SIZE2* farScrSize,
    ANGLE* angle, VECTOR2D aspectRatio
)
{
    // define screen size
    (*nearScrSize).width = tan((*angle).horiz / 2 * PI / 180) * nearZ * 2;
    (*nearScrSize).height = (*nearScrSize).width * aspectRatio.y / aspectRatio.x;

    (*angle).vert = atan2((*nearScrSize).height / 2, nearZ) * 180 / PI * 2;

    (*farScrSize).width = (*nearScrSize).width / 2 * farZ / nearZ;
    (*farScrSize).height = (*farScrSize).width * aspectRatio.y / aspectRatio.x;
    
    // Define coordinates of view area vertices on xz axis
    pointXZ[VP1].x = -(*nearScrSize).width / 2;
    pointXZ[VP2].x = -(*farScrSize).width / 2;
    pointXZ[VP3].x = (*farScrSize).width / 2;
    pointXZ[VP4].x = (*nearScrSize).width / 2;

    pointXZ[VP1].z = -nearZ;
    pointXZ[VP2].z = -farZ;
    pointXZ[VP3].z = -farZ;
    pointXZ[VP4].z = -nearZ;

    // Define coordinates of view area vertices on yz axis
    pointYZ[VP1].y = (*nearScrSize).height / 2;
    pointYZ[VP2].y = (*farScrSize).height / 2;
    pointYZ[VP3].y = -(*farScrSize).height / 2;
    pointYZ[VP4].y = -(*nearScrSize).height / 2;

    pointYZ[VP1].z = -nearZ;
    pointYZ[VP2].z = -farZ;
    pointYZ[VP3].z = -farZ;
    pointYZ[VP4].z = -nearZ;

    // Enter 3D camera coordinates for view volume
    point3D[RECT_FRONT_TOP_LEFT].x = pointXZ[VP1].x;
    point3D[RECT_FRONT_TOP_LEFT].y = pointYZ[VP1].y;
    point3D[RECT_FRONT_TOP_LEFT].z = -nearZ;

    point3D[RECT_FRONT_TOP_RIGHT].x = pointXZ[VP4].x;
    point3D[RECT_FRONT_TOP_RIGHT].y = pointYZ[VP1].y;
    point3D[RECT_FRONT_TOP_RIGHT].z = -nearZ;

    point3D[RECT_FRONT_BOTTOM_RIGHT].x = pointXZ[VP4].x;
    point3D[RECT_FRONT_BOTTOM_RIGHT].y = pointYZ[VP4].y;
    point3D[RECT_FRONT_BOTTOM_RIGHT].z = -nearZ;

    point3D[RECT_FRONT_BOTTOM_LEFT].x = pointXZ[VP1].x;
    point3D[RECT_FRONT_BOTTOM_LEFT].y = pointYZ[VP4].y;
    point3D[RECT_FRONT_BOTTOM_LEFT].z = -nearZ;

    point3D[RECT_BACK_TOP_LEFT].x = pointXZ[VP2].x;
    point3D[RECT_BACK_TOP_LEFT].y = pointYZ[VP2].y;
    point3D[RECT_BACK_TOP_LEFT].z = -farZ;

    point3D[RECT_BACK_TOP_RIGHT].x = pointXZ[VP3].x;
    point3D[RECT_BACK_TOP_RIGHT].y = pointYZ[VP2].y;
    point3D[RECT_BACK_TOP_RIGHT].z = -farZ;

    point3D[RECT_BACK_BOTTOM_RIGHT].x = pointXZ[VP3].x;
    point3D[RECT_BACK_BOTTOM_RIGHT].y = pointYZ[VP3].y;
    point3D[RECT_BACK_BOTTOM_RIGHT].z = -farZ;

    point3D[RECT_BACK_BOTTOM_LEFT].x = pointXZ[VP2].x;
    point3D[RECT_BACK_BOTTOM_LEFT].y = pointYZ[VP3].y;
    point3D[RECT_BACK_BOTTOM_LEFT].z = -farZ;

    // Enter a point on the surface
    face[SURFACE_TOP].oneV = point3D[RECT_FRONT_TOP_LEFT];
    face[SURFACE_FRONT].oneV = point3D[RECT_FRONT_TOP_LEFT];
    face[SURFACE_RIGHT].oneV = point3D[RECT_BACK_BOTTOM_RIGHT];
    face[SURFACE_LEFT].oneV = point3D[RECT_FRONT_TOP_LEFT];
    face[SURFACE_BACK].oneV = point3D[RECT_BACK_BOTTOM_RIGHT];
    face[SURFACE_BOTTOM].oneV = point3D[RECT_BACK_BOTTOM_RIGHT];

    // Enter the starting and ending points of the line segments of the view volume
    lineStartPoint[RECT_L1] = point3D[RECT_L1_STARTPT];
    lineStartPoint[RECT_L2] = point3D[RECT_L2_STARTPT];
    lineStartPoint[RECT_L3] = point3D[RECT_L3_STARTPT];
    lineStartPoint[RECT_L4] = point3D[RECT_L4_STARTPT];
    lineStartPoint[RECT_L5] = point3D[RECT_L5_STARTPT];
    lineStartPoint[RECT_L6] = point3D[RECT_L6_STARTPT];
    lineStartPoint[RECT_L7] = point3D[RECT_L7_STARTPT];
    lineStartPoint[RECT_L8] = point3D[RECT_L8_STARTPT];
    lineStartPoint[RECT_L9] = point3D[RECT_L9_STARTPT];
    lineStartPoint[RECT_L10] = point3D[RECT_L10_STARTPT];
    lineStartPoint[RECT_L11] = point3D[RECT_L11_STARTPT];
    lineStartPoint[RECT_L12] = point3D[RECT_L12_STARTPT];

    lineEndPoint[RECT_L1] = point3D[RECT_L1_ENDPT];
    lineEndPoint[RECT_L2] = point3D[RECT_L2_ENDPT];
    lineEndPoint[RECT_L3] = point3D[RECT_L3_ENDPT];
    lineEndPoint[RECT_L4] = point3D[RECT_L4_ENDPT];
    lineEndPoint[RECT_L5] = point3D[RECT_L5_ENDPT];
    lineEndPoint[RECT_L6] = point3D[RECT_L6_ENDPT];
    lineEndPoint[RECT_L7] = point3D[RECT_L7_ENDPT];
    lineEndPoint[RECT_L8] = point3D[RECT_L8_ENDPT];
    lineEndPoint[RECT_L9] = point3D[RECT_L9_ENDPT];
    lineEndPoint[RECT_L10] = point3D[RECT_L10_ENDPT];
    lineEndPoint[RECT_L11] = point3D[RECT_L11_ENDPT];
    lineEndPoint[RECT_L12] = point3D[RECT_L12_ENDPT];

    vec.minusVec3d(lineStartPoint, lineEndPoint);

    lineVec = vec.resultVector3D;

    std::vector<VECTOR3D> calcLineVecA(6);
    std::vector<VECTOR3D> calcLineVecB(6);

    calcLineVecA[SURFACE_TOP] = lineVec[RECT_L1];
    calcLineVecB[SURFACE_TOP] = lineVec[RECT_L5];

    calcLineVecA[SURFACE_FRONT] = lineVec[RECT_L1];
    calcLineVecB[SURFACE_FRONT] = lineVec[RECT_L2];

    calcLineVecA[SURFACE_RIGHT] = lineVec[RECT_L2];
    calcLineVecB[SURFACE_RIGHT] = lineVec[RECT_L6];

    calcLineVecA[SURFACE_LEFT] = lineVec[RECT_L4];
    calcLineVecB[SURFACE_LEFT] = lineVec[RECT_L5];

    calcLineVecA[SURFACE_BACK] = lineVec[RECT_L9];
    calcLineVecB[SURFACE_BACK] = lineVec[RECT_L10];

    calcLineVecA[SURFACE_BOTTOM] = lineVec[RECT_L3];
    calcLineVecB[SURFACE_BOTTOM] = lineVec[RECT_L7];

    vec.crossProduct(calcLineVecA, calcLineVecB);

    double calcInSqrt;
    for (int i = 0; i < face.size(); ++i)
    {
        calcInSqrt 
        = pow(vec.resultVector3D[i].x, 2) + pow(vec.resultVector3D[i].y, 2) + pow(vec.resultVector3D[i].z, 2);
        face[i].normal.x = vec.resultVector3D[i].x / abs(sqrt(calcInSqrt));
        face[i].normal.y = vec.resultVector3D[i].y / abs(sqrt(calcInSqrt));
        face[i].normal.z = vec.resultVector3D[i].z / abs(sqrt(calcInSqrt));
    }
}

int ViewVolume::clipRange(std::vector<OBJ_FILE> objData, std::vector<double> degree, ANGLE angle, int loopI)
{
    // Z-axis direction determination
    if (objData[loopI].range.origin.z > pointXZ[VP2].z && objData[loopI].range.opposite.z < pointXZ[VP1].z)
    {
        // X-axis direction determination
        if 
        (
            // ORIGIN
            degree[loopI*4 + 0] <= -90 + angle.horiz / 2 &&

            // OPPOSITE
            degree[loopI*4 + 2] >= -90 -angle.horiz / 2
        )
        {
            // Y-axis direction determination
            if
            (
                // ORIGIN
                degree[loopI*4 + 1] <= -90 + angle.vert / 2  &&

                // OPPOSITE
                degree[loopI*4 + 3] >= -90 - angle.vert / 2
            )
            {
                return loopI;
            }
        }
    }
    return NULL_INDEX;
}

bool ViewVolume::clipV(double v, double degreeXZ, double degreeZY, ANGLE angle, int loopI)
{
    if (v > pointXZ[VP2].z && v < pointXZ[VP1].z)
    {
        // X-axis direction determination
        if 
        (degreeXZ <= -90 + angle.horiz / 2 && degreeXZ >= -90 -angle.horiz / 2)
        {
            // Y-axis direction determination
            if(degreeZY <= -90 + angle.vert / 2 && degreeZY >= -90 - angle.vert / 2)
            {
                return  true;
            }
        }
    }
    return false;
}