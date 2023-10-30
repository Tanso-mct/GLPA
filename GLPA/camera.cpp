#define DEBUG_CAMERA_
#include "camera.h"

void VIEWVOLUME::define
(
    double nearZ, double farZ,
    SIZE2 nearScrPxSize, SIZE2 farScrPxSize,
    ANGLE* angle, VECTOR2D aspectRatio
)
{
    // define screen size
    nearScrPxSize.width = tan((*angle).horiz / 2 * PI / 180) * nearZ * 2;
    nearScrPxSize.height = nearScrPxSize.width * aspectRatio.y / aspectRatio.x;

    (*angle).vert = atan2(nearScrPxSize.height / 2, nearZ) * 180 / PI * 2;

    farScrPxSize.width = nearScrPxSize.width / 2 * farZ / nearZ;
    farScrPxSize.height = farScrPxSize.width * aspectRatio.y / aspectRatio.x;
    
    // Define coordinates of view area vertices on xz axis
    pointXZ[VP1].x = -nearScrPxSize.width / 2;
    pointXZ[VP2].x = -farScrPxSize.width / 2;
    pointXZ[VP3].x = farScrPxSize.width / 2;
    pointXZ[VP4].x = nearScrPxSize.width / 2;

    pointXZ[VP1].z = -nearZ;
    pointXZ[VP2].z = -farZ;
    pointXZ[VP3].z = -farZ;
    pointXZ[VP4].z = -nearZ;

    // Define coordinates of view area vertices on yz axis
    pointYZ[VP1].y = nearScrPxSize.height / 2;
    pointYZ[VP2].y = farScrPxSize.height / 2;
    pointYZ[VP3].y = -farScrPxSize.height / 2;
    pointYZ[VP4].y = -nearScrPxSize.height / 2;

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
    face[SURFACE_TOP].oneV = point3D[0];
    face[SURFACE_FRONT].oneV = point3D[0];
    face[SURFACE_RIGHT].oneV = point3D[6];
    face[SURFACE_LEFT].oneV = point3D[0];
    face[SURFACE_BACK].oneV = point3D[6];
    face[SURFACE_BOTTOM].oneV = point3D[6];

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
        calcInSqrt = pow(vec.resultVector3D[i].x, 2) + pow(vec.resultVector3D[i].y, 2) + pow(vec.resultVector3D[i].z, 2);
        face[i].normal.x = vec.resultVector3D[i].x / abs(sqrt(calcInSqrt));
        face[i].normal.y = vec.resultVector3D[i].y / abs(sqrt(calcInSqrt));
        face[i].normal.z = vec.resultVector3D[i].z / abs(sqrt(calcInSqrt));
    }
}

void CAMERA::coordinateTransRange(std::vector<OBJ_FILE>* objData)
{
    // origin and opposite point data
    std::vector<VECTOR3D> pointData;

    for (int i = 0; i < (*objData).size(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            pointData.push_back((*objData)[i].range.wVertex[j]);
        }
    }

    vec.posTrans(pointData, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);

    for (int i = 0; i < (*objData).size(); ++i)
    {
        (*objData)[i].range.origin = mtx.resultMatrices[i*8];
        (*objData)[i].range.opposite = mtx.resultMatrices[i*8];
    }
    
    for (int i = 0; i < (*objData).size(); ++i)
    {
        for (int j = 1; j < 8; ++j)
        {
            // Processing with respect to origin point
            if ((*objData)[i].range.origin.x > mtx.resultMatrices[i*8 + j].x)
            {   
                (*objData)[i].range.origin.x = mtx.resultMatrices[i*8 + j].x;
            }
            if ((*objData)[i].range.origin.y > mtx.resultMatrices[i*8 + j].y)
            {
                (*objData)[i].range.origin.y = mtx.resultMatrices[i*8 + j].y;
            }
            if ((*objData)[i].range.origin.z < mtx.resultMatrices[i*8 + j].z)
            {
                (*objData)[i].range.origin.z = mtx.resultMatrices[i*8 + j].z;
            }

            // Processing with respect to opposite point
            if ((*objData)[i].range.opposite.x < mtx.resultMatrices[i*8 + j].x)
            {
                (*objData)[i].range.opposite.x = mtx.resultMatrices[i*8 + j].x;
            }
            if ((*objData)[i].range.opposite.y < mtx.resultMatrices[i*8 + j].y)
            {
                (*objData)[i].range.opposite.y = mtx.resultMatrices[i*8 + j].y;
            }
            if ((*objData)[i].range.opposite.z > mtx.resultMatrices[i*8 + j].z)
            {
                (*objData)[i].range.opposite.z = mtx.resultMatrices[i*8 + j].z;
            }
        }
    }
    
}

void CAMERA::clippingRange(std::vector<OBJ_FILE> objData)
{
    std::vector<double> vertValue;
    std::vector<double> horizValue;

    vertValue.resize(objData.size() * 4);
    horizValue.resize(objData.size() * 4);

    for (int i = 0; i < objData.size(); ++i)
    {
        // origin XZ
        vertValue[i*4 + 0] = objData[i].range.opposite.z;
        horizValue[i*4 + 0] = objData[i].range.origin.x;

        // origin YZ
        vertValue[i*4 + 1] = objData[i].range.opposite.z;
        horizValue[i*4 + 1] = objData[i].range.origin.y;

        // opposite XZ
        vertValue[i*4 + 2] = objData[i].range.opposite.z;
        horizValue[i*4 + 2] = objData[i].range.opposite.x;

        // opposite YZ
        vertValue[i*4 + 3] = objData[i].range.opposite.z;
        horizValue[i*4 + 3] = objData[i].range.opposite.y;
    }

    tri.get2dVecAngle(vertValue, horizValue);
    withinRangeAryNum.resize(0);
    for (int i = 0; i < objData.size(); ++i)
    {
        // Z-axis direction determination
        if 
        (
            objData[i].range.origin.z > viewPointXZ[VP2].z &&
            objData[i].range.opposite.z < viewPointXZ[VP1].z
        )
        {
            // X-axis direction determination
            if 
            (
                // ORIGIN
                tri.resultDegree[i*4 + 0] <= -90 + horizAngle / 2 &&

                // OPPOSITE
                tri.resultDegree[i*4 + 2] >= -90 -horizAngle / 2
            )
            {
                // Y-axis direction determination
                if
                (
                    // ORIGIN
                    tri.resultDegree[i*4 + 1] <= -90 + vertAngle / 2  &&

                    // OPPOSITE
                    tri.resultDegree[i*4 + 3] >= -90 - vertAngle / 2
                )
                {
                    withinRangeAryNum.push_back(i);
                }
            }
        }
    }
}

void CAMERA::polyBilateralJudge(std::vector<OBJ_FILE> objData)
{
    // Stores the number of faces of each object in the clipping area
    std::vector<int> faceAmout;

    // Stores the coordinates of the vertices of the face corresponding to the normal vector of the face
    std::vector<VECTOR3D> planeVertex;

    // Stores the normals of the faces of objects in the clipping area
    std::vector<VECTOR3D> planeNormal;

    // Stores all normal vectors and vertex world cooridinate of the surface to be calculated
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < objData[withinRangeAryNum[i]].poly.normal.size(); ++j)
        {
            planeVertex.push_back
            (
                objData[withinRangeAryNum[i]].v.world
                [
                    objData[withinRangeAryNum[i]].poly.v[j].num1
                ]
            );

            planeNormal.push_back
            (
                objData[withinRangeAryNum[i]].v.normal
                [
                    objData[withinRangeAryNum[i]].poly.normal[j].num1
                ]
            );
        }
        faceAmout.push_back(objData[withinRangeAryNum[i]].poly.normal.size());
    }

    // Camera coordinate transformation of the vertices of a face
    vec.posTrans(planeVertex, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);
    planeVertex = mtx.resultMatrices;

    // Camera coordinate transformation of the normal vector of a surface
    mtx.rotTrans(planeNormal, rotAngle);
    planeNormal = mtx.resultMatrices;

    vec.dotProduct(planeNormal, planeVertex);

    // Stores which polygons of which objects are facing front
    int sumFaceAmout = 0;
    numPolyFacing.resize(0);
    numPolyFacing.resize(withinRangeAryNum.size());
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < faceAmout[i]; ++j)
        {
            if (vec.resultVector[sumFaceAmout + j] < 0)
            {
                numPolyFacing[i].push_back(j);
            }
        }
        sumFaceAmout += faceAmout[i];
    }
}

void CAMERA::coordinateTrans(std::vector<OBJ_FILE> objData)
{
    // Stores all vertices of surface polygons
    polyVertex.resize(0);
    polyNormal.resize(0);
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < numPolyFacing[i].size(); ++j)
        {
            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i][j]].num1
                ]
            );

            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i][j]].num2
                ]
            );

            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i][j]].num3
                ]   
            );

            polyNormal.push_back
            (
                objData[i].v.normal
                [
                    objData[i].poly.normal[numPolyFacing[i][j]].num1
                ]   
            );
        }
    }

    // Camera coordinate transformation of vertex data
    vec.posTrans(polyVertex, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);
    polyVertex = mtx.resultMatrices;

    mtx.rotTrans(polyNormal, rotAngle);
    polyNormal = mtx.resultMatrices;
}

std::vector<bool> CAMERA::vertexInViewVolume(std::vector<VECTOR3D> v)
{
    std::vector<double> vertValue;
    std::vector<double> horizValue;

    vertValue.resize(v.size() * 2);
    horizValue.resize(v.size() * 2);

    for (int i = 0; i < v.size(); ++i)
    {
        // XZ
        vertValue[i*2 + 0] = v[i].z;
        horizValue[i*2 + 0] = v[i].x;

        // YZ
        vertValue[i*2 + 1] = v[i].z;
        horizValue[i*2 + 1] = v[i].y;
    }

    tri.get2dVecAngle(vertValue, horizValue);

    std::vector<bool> vInViewVolume;
    vInViewVolume.resize(v.size(), false);
    // Z-axis direction determination
    for (int i = 0; i < v.size(); ++i)
    {
        if 
        (
            v[i].z > viewPointXZ[VP2].z &&
            v[i].z < viewPointXZ[VP1].z
        )
        {
            // X-axis direction determination
            if 
            (tri.resultDegree[i*2 + 0] <= -90 + horizAngle / 2 && tri.resultDegree[i*2 + 0] >= -90 -horizAngle / 2)
            {
                // Y-axis direction determination
                if(tri.resultDegree[i*2 + 1] <= -90 + vertAngle / 2 && tri.resultDegree[i*2 + 1] >= -90 - vertAngle / 2)
                {
                    vInViewVolume[i] = true;
                }
            }
        }
    }
    return vInViewVolume;
}

std::vector<std::vector<int>> CAMERA::clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> rangePoly, int processObjectAmout)
{
    std::vector<std::vector<int>> indexInViewVolumePoly;
    indexInViewVolumePoly.resize(processObjectAmout);

    std::vector<double> vertValue;
    std::vector<double> horizValue;

    vertValue.resize(rangePoly.size() * 4);
    horizValue.resize(rangePoly.size() * 4);

    for (int i = 0; i < rangePoly.size(); ++i)
    {
        for (int j = 0; j < rangePoly[i].size(); ++j)
        {
            // origin XZ
            vertValue[i*4 + 0] = rangePoly[i][j].opposite.z;
            horizValue[i*4 + 0] = rangePoly[i][j].origin.x;

            // origin YZ
            vertValue[i*4 + 1] = rangePoly[i][j].opposite.z;
            horizValue[i*4 + 1] = rangePoly[i][j].origin.y;

            // opposite XZ
            vertValue[i*4 + 2] = rangePoly[i][j].opposite.z;
            horizValue[i*4 + 2] = rangePoly[i][j].opposite.x;

            // opposite YZ
            vertValue[i*4 + 3] = rangePoly[i][j].opposite.z;
            horizValue[i*4 + 3] = rangePoly[i][j].opposite.y;
        }
    }

    tri.get2dVecAngle(vertValue, horizValue);
    for (int i = 0; i < processObjectAmout; ++i)
    {
        for (int j = 0; j < rangePoly[i].size(); ++j)
        {
            // Z-axis direction determination
            if 
            (
                rangePoly[i][j].origin.z > viewPointXZ[VP2].z &&
                rangePoly[i][j].opposite.z < viewPointXZ[VP1].z
            )
            {
                // X-axis direction determination
                if 
                (
                    // ORIGIN
                    tri.resultDegree[i*4 + 0] <= -90 + horizAngle / 2 &&

                    // OPPOSITE
                    tri.resultDegree[i*4 + 2] >= -90 -horizAngle / 2
                )
                {
                    // Y-axis direction determination
                    if
                    (
                        // ORIGIN
                        tri.resultDegree[i*4 + 1] <= -90 + vertAngle / 2  &&

                        // OPPOSITE
                        tri.resultDegree[i*4 + 3] >= -90 - vertAngle / 2
                    )
                    {
                        indexInViewVolumePoly[i].push_back(j);
                    }
                }
            }
        }
    }

    return indexInViewVolumePoly;
}


void CAMERA::polyInViewVolumeJudge(std::vector<OBJ_FILE> objData)
{
    // Stores the vertices that make up the line segment
    std::vector<VECTOR3D> polyLineVA;
    std::vector<VECTOR3D> polyLineVB;

    numPolyInViewVolume.resize(0);
    numPolyInViewVolume.resize(withinRangeAryNum.size());
    numPolyExitsIViewVolume.resize(0);
    numPolyExitsIViewVolume.resize(withinRangeAryNum.size());
    clippedPolyVertex.resize(0);
    clippedPolyVertex.resize(withinRangeAryNum.size());
    
    // If one of the vertices is in the view volume, 
    // it is excluded from the intersection determination as a drawing target.
    int polySum = 0;
    int inVolumeAmoutV = 0;
    std::vector<std::vector<int>> indexNumPolyFacing;
    indexNumPolyFacing.resize(withinRangeAryNum.size());
    
    std::vector<bool> vInViewVolume = vertexInViewVolume(polyVertex);
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        numPolyInViewVolume[i].resize(numPolyFacing[i].size(), -1);
        clippedPolyVertex[i].resize(numPolyFacing[i].size()*2);
        for (int j = 0; j < numPolyFacing[i].size(); ++j)
        {
            if (vInViewVolume[polySum + j*3 + 0])
            {
                clippedPolyVertex[i][j*2].push_back(polyVertex[polySum + j*3 + 0]);
                inVolumeAmoutV += 1;
            }
            if (vInViewVolume[polySum + j*3 + 1])
            {
                clippedPolyVertex[i][j*2].push_back(polyVertex[polySum + j*3 + 1]);
                inVolumeAmoutV += 1;
            }
            if (vInViewVolume[polySum + j*3 + 2])
            {
                clippedPolyVertex[i][j*2].push_back(polyVertex[polySum + j*3 + 2]);
                inVolumeAmoutV += 1;
            }

            if (inVolumeAmoutV == 3)
            {
                numPolyInViewVolume[i][j] = numPolyFacing[i][j];
            }
            else
            {
                numPolyExitsIViewVolume[i].push_back(numPolyFacing[i][j]);
                indexNumPolyFacing[i].push_back(j);
                // Input vertex A
                polyLineVA.push_back
                (
                    polyVertex[polySum + j*3 + 0]
                );

                polyLineVA.push_back
                (
                    polyVertex[polySum + j*3 + 1]
                );

                polyLineVA.push_back
                (
                    polyVertex[polySum + j*3 + 2]
                );

                // Input vertex B
                polyLineVB.push_back
                (
                    polyVertex[polySum + j*3 + 1]
                );

                polyLineVB.push_back
                (
                    polyVertex[polySum + j*3 + 2]
                );

                polyLineVB.push_back
                (
                    polyVertex[polySum + j*3 + 0]
                );
            }

            inVolumeAmoutV = 0;
        }
        polySum += numPolyFacing[i].size() * 3;
    }

    // Obtaining the coordinates of the intersection of a line consisting of the vertices of a triangle 
    // with the face of the view volumeA
    eq.getLinePlaneI
    (
        polyLineVA,
        polyLineVB,
        viewVolumeFaceVertex,
        viewVolumeFaceNormal
    );

    // Determine if the requested intersection is on the view volume plane
    std::vector<VECTOR3D> viewVolumePoint1;
    std::vector<VECTOR3D> viewVolumePoint2;

    viewVolumePoint1.resize(12);
    viewVolumePoint2.resize(12);

    // Starting point of the front side
    viewVolumePoint1[0] = viewPoint[0];
    viewVolumePoint1[1] = viewPoint[1];
    viewVolumePoint1[2] = viewPoint[2];
    viewVolumePoint1[3] = viewPoint[3];

    // Starting point of the side connecting the front and back surfaces
    viewVolumePoint1[4] = viewPoint[0];
    viewVolumePoint1[5] = viewPoint[1];
    viewVolumePoint1[6] = viewPoint[2];
    viewVolumePoint1[7] = viewPoint[3];

    // Starting point of the rear edge
    viewVolumePoint1[8] = viewPoint[4];
    viewVolumePoint1[9] = viewPoint[5];
    viewVolumePoint1[10] = viewPoint[6];
    viewVolumePoint1[11] = viewPoint[7];

    // Ending point of the front side
    viewVolumePoint2[0] = viewPoint[1];
    viewVolumePoint2[1] = viewPoint[2];
    viewVolumePoint2[2] = viewPoint[3];
    viewVolumePoint2[3] = viewPoint[0];

    // Ending point of the side connecting the front and back surfaces
    viewVolumePoint2[4] = viewPoint[4];
    viewVolumePoint2[5] = viewPoint[5];
    viewVolumePoint2[6] = viewPoint[6];
    viewVolumePoint2[7] = viewPoint[7];

    // Ending point of the rear edge
    viewVolumePoint2[8] = viewPoint[5];
    viewVolumePoint2[9] = viewPoint[6];
    viewVolumePoint2[10] = viewPoint[7];
    viewVolumePoint2[11] = viewPoint[4];

    vec.minusVec3d(viewVolumePoint1, viewVolumePoint2);
    std::vector<VECTOR3D> vecViewVolumeLine = vec.resultVector3D;

    // Stores view volume line vectors
    std::vector<VECTOR3D> calcViewVolumeLine;

    viewVolumePoint1.resize(0);
    std::vector<VECTOR3D> viewVolumeFaceI;

    polySum = 0;
    int linePlaneIindex = 0;
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < numPolyExitsIViewVolume[i].size(); ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < viewVolumeFaceNormal.size(); ++l)
                {   
                    if (eq.existenceI[polySum][l] == I_TRUE)
                    {
                        // Top
                        if (l == 0)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[0]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[5]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[8]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[4]);

                            viewVolumePoint1.push_back(viewPoint[0]);
                            viewVolumePoint1.push_back(viewPoint[1]);
                            viewVolumePoint1.push_back(viewPoint[4]);
                            viewVolumePoint1.push_back(viewPoint[0]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }
                        // Front
                        else if (l == 1)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[0]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[1]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[2]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[3]);

                            viewVolumePoint1.push_back(viewPoint[0]);
                            viewVolumePoint1.push_back(viewPoint[1]);
                            viewVolumePoint1.push_back(viewPoint[2]);
                            viewVolumePoint1.push_back(viewPoint[3]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }
                        // Right
                        else if (l == 2)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[5]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[9]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[6]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[1]);

                            viewVolumePoint1.push_back(viewPoint[1]);
                            viewVolumePoint1.push_back(viewPoint[5]);
                            viewVolumePoint1.push_back(viewPoint[2]);
                            viewVolumePoint1.push_back(viewPoint[1]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }
                        // Left
                        else if (l == 3)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[4]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[11]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[7]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[3]);

                            viewVolumePoint1.push_back(viewPoint[0]);
                            viewVolumePoint1.push_back(viewPoint[7]);
                            viewVolumePoint1.push_back(viewPoint[3]);
                            viewVolumePoint1.push_back(viewPoint[3]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }
                        // Back
                        else if (l == 4)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[8]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[9]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[10]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[11]);

                            viewVolumePoint1.push_back(viewPoint[4]);
                            viewVolumePoint1.push_back(viewPoint[5]);
                            viewVolumePoint1.push_back(viewPoint[6]);
                            viewVolumePoint1.push_back(viewPoint[7]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }
                        // Bottom
                        else if (l == 5)
                        {
                            calcViewVolumeLine.push_back(vecViewVolumeLine[10]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[6]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[2]);
                            calcViewVolumeLine.push_back(vecViewVolumeLine[7]);

                            viewVolumePoint1.push_back(viewPoint[6]);
                            viewVolumePoint1.push_back(viewPoint[2]);
                            viewVolumePoint1.push_back(viewPoint[2]);
                            viewVolumePoint1.push_back(viewPoint[3]);

                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                            viewVolumeFaceI.push_back(eq.linePlaneI[linePlaneIindex]);
                        }

                        linePlaneIindex += 1;
                    }
                }
                polySum += 1;
            }
        }
    }

    vec.minusVec3d(viewVolumePoint1, viewVolumeFaceI);
    std::vector<VECTOR3D> calcViewVolumeI = vec.resultVector3D;

    vec.crossProduct(calcViewVolumeLine, calcViewVolumeI);

    std::vector<VECTOR3D> calcDotOnViewVolumeI1;
    std::vector<VECTOR3D> calcDotOnViewVolumeI2;

    for (int i = 0; i < vec.resultVector3D.size() / 4; ++i)
    {
        calcDotOnViewVolumeI1.push_back(vec.resultVector3D[i*4 + 0]);
        calcDotOnViewVolumeI1.push_back(vec.resultVector3D[i*4 + 0]);
        calcDotOnViewVolumeI1.push_back(vec.resultVector3D[i*4 + 0]);

        calcDotOnViewVolumeI2.push_back(vec.resultVector3D[i*4 + 1]);
        calcDotOnViewVolumeI2.push_back(vec.resultVector3D[i*4 + 2]);
        calcDotOnViewVolumeI2.push_back(vec.resultVector3D[i*4 + 3]);
    }

    vec.dotProduct(calcDotOnViewVolumeI1, calcDotOnViewVolumeI2);
    std::vector<double> dotoOnViewVolumeI = vec.resultVector;

    numPolyAllVLINENotInViewVolume.resize(0);
    numPolyAllVLINENotInViewVolume.resize(withinRangeAryNum.size());

    polySum = 0;
    linePlaneIindex = 0;
    int amoutPolyFacing = 0;
    bool findTrueI = false;
    numPolyTrueIViewVolume.resize(withinRangeAryNum.size());

    // Find and store the value of one vertex and normal vector of a polygon outside the view volume 
    // for all three lines and all three points and polygons with intersections
    std::vector<VECTOR3D> polyPlaneVertex;
    std::vector<VECTOR3D> polyPlaneNormal;

    // Store intersection coordinates for each polygon, if any
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < numPolyExitsIViewVolume[i].size(); ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < viewVolumeFaceNormal.size(); ++l)
                {   
                    if (eq.existenceI[polySum][l] == I_TRUE)
                    {
                        if 
                        (
                            dotoOnViewVolumeI[linePlaneIindex*3 + 0] < 0 && 
                            dotoOnViewVolumeI[linePlaneIindex*3 + 1] < 0 &&
                            dotoOnViewVolumeI[linePlaneIindex*3 + 2] < 0
                        )
                        {
                            numPolyInViewVolume[i][indexNumPolyFacing[i][j]] = numPolyExitsIViewVolume[i][j];
                            clippedPolyVertex[i][indexNumPolyFacing[i][j]*2 + 1].push_back(eq.linePlaneI[linePlaneIindex]);
                            findTrueI = true;
                        }
                        linePlaneIindex += 1;
                    }
                }
                polySum += 1;
            }

            // Stores polygon numbers without a single intersection
            if (!findTrueI)
            {
                numPolyAllVLINENotInViewVolume[i].push_back(j);
            }
            else
            {
                numPolyTrueIViewVolume[i].push_back(j);
                polyPlaneVertex.push_back
                (
                    polyVertex[amoutPolyFacing + indexNumPolyFacing[i][j]*VECTOR3]
                );

                polyPlaneNormal.push_back
                (
                    polyNormal[amoutPolyFacing / 3 + indexNumPolyFacing[i][j]]
                );
            }
            findTrueI = false;
        }
        amoutPolyFacing += numPolyFacing[i].size() * 3;
    }

    // Polygons that are outside the view volume at all three points and all three sides are converted to RANGE_CUBA
    std::vector<std::vector<RANGE_CUBE_POLY>> rangePolyAllOutside;

    amoutPolyFacing = 0;

    rangePolyAllOutside.resize(withinRangeAryNum.size());
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        rangePolyAllOutside[i].resize(numPolyAllVLINENotInViewVolume[i].size());
        for (int j = 0; j < numPolyAllVLINENotInViewVolume[i].size(); ++j)
        {
            rangePolyAllOutside[i][j].origin 
            = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 0];

            rangePolyAllOutside[i][j].opposite 
            = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 0];

            // origin
            // num2
            if (
                rangePolyAllOutside[i][j].origin.x 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].x
            )
            {
                rangePolyAllOutside[i][j].origin.x 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].x;
            }
            if (
                rangePolyAllOutside[i][j].origin.y 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].y
            )
            {
                rangePolyAllOutside[i][j].origin.y 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].y;
            }
            if (
                rangePolyAllOutside[i][j].origin.z 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].z
            )
            {
                rangePolyAllOutside[i][j].origin.z 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].z;
            }

            // num3
            if (
                rangePolyAllOutside[i][j].origin.x 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].x
            )
            {
                rangePolyAllOutside[i][j].origin.x 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].x;
            }
            if (
                rangePolyAllOutside[i][j].origin.y 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].y
            )
            {
                rangePolyAllOutside[i][j].origin.y 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].y;
            }
            if (
                rangePolyAllOutside[i][j].origin.z 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].z
            )
            {
                rangePolyAllOutside[i][j].origin.z 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].z;
            }


            // opposite
            // num2
            if (
                rangePolyAllOutside[i][j].opposite.x 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].x
            )
            {
                rangePolyAllOutside[i][j].opposite.x 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].x;
            }
            if (
                rangePolyAllOutside[i][j].opposite.y 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].y
            )
            {
                rangePolyAllOutside[i][j].opposite.y 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].y;
            }
            if (
                rangePolyAllOutside[i][j].opposite.z 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].z
            )
            {
                rangePolyAllOutside[i][j].opposite.z 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 1].z;
            }

            // num3
            if (
                rangePolyAllOutside[i][j].opposite.x 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].x
            )
            {
                rangePolyAllOutside[i][j].opposite.x 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].x;
            }
            if (
                rangePolyAllOutside[i][j].opposite.y 
                < polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].y
            )
            {
                rangePolyAllOutside[i][j].opposite.y 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].y;
            }
            if (
                rangePolyAllOutside[i][j].opposite.z 
                > polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].z
            )
            {
                rangePolyAllOutside[i][j].opposite.z 
                = polyVertex[amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][j]*VECTOR3 + 2].z;
            }
        }
        amoutPolyFacing += numPolyFacing[i].size() * 3;
    }   

    // If the polygon range intersects the view volume, store the polygon number with its index number
    // This index number refers to the number of the numPollyAllVLINENotInViewVolume
    std::vector<std::vector<int>> indexInViewVolumeAllOutside = clippingRange(rangePolyAllOutside, withinRangeAryNum.size());

    amoutPolyFacing = 0;

    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < indexInViewVolumeAllOutside[i].size(); ++j)
        {
            polyPlaneVertex.push_back
            (
                polyVertex
                [amoutPolyFacing + numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][j]]*VECTOR3]
            );

            polyPlaneNormal.push_back
            (
                polyNormal
                [amoutPolyFacing / 3 + numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][j]]]
            );
        }
        amoutPolyFacing += numPolyFacing[i].size() * 3;
    }

    // Stores the start and end points of the 12 lines of the view volume
    std::vector<VECTOR3D> viewVolumeLineA;
    std::vector<VECTOR3D> viewVolumeLineB;

    for (int i = 0; i < 4; ++i)
    {
        if (i == 3)
        {
            viewVolumeLineA.push_back(viewPoint[i]);
            viewVolumeLineB.push_back(viewPoint[0]);
        }
        else
        {
            viewVolumeLineA.push_back(viewPoint[i]);
            viewVolumeLineB.push_back(viewPoint[i + 1]);
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        viewVolumeLineA.push_back(viewPoint[i]);
        viewVolumeLineB.push_back(viewPoint[i + 4]);
    }

    for (int i = 4; i < 8; ++i)
    {
        if (i == 7)
        {
            viewVolumeLineA.push_back(viewPoint[i]);
            viewVolumeLineB.push_back(viewPoint[4]);
        }
        else
        {
            viewVolumeLineA.push_back(viewPoint[i]);
            viewVolumeLineB.push_back(viewPoint[i + 1]);
        }
    }

    // Obtain intersection with view volume edges
    eq.getLinePlaneI
    (
        viewVolumeLineA,
        viewVolumeLineB,
        polyPlaneVertex,
        polyPlaneNormal
    );

    // After determining whether the intersection is inside a polygon or not, if it is inside, 
    // it is stored in ClippedPolyVertex.
    polySum = 0;
    linePlaneIindex = 0;

    // Stores information on polygons with intersections to find cross products
    std::vector<VECTOR3D> calcPolyVertex;
    std::vector<VECTOR3D> calcPolyVertexTo;
    std::vector<VECTOR3D> calcPolySurfaceIpoint;
    std::vector<POLYSURFACE_I_INFO> polySurfaceIinfo;
    POLYSURFACE_I_INFO useInLoopPolySurfaceIinfo;
    for (int j = 0; j < viewVolumeLineA.size(); ++j)
    {
        for (int i = 0; i < withinRangeAryNum.size(); ++i)
        {
            for (int k = 0; k < numPolyTrueIViewVolume[i].size(); ++k)
            {
                if (eq.existenceI[j][polySum + k] == I_TRUE)
                {
                    calcPolyVertex.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 0]
                    );
                    calcPolyVertex.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 1]
                    );
                    calcPolyVertex.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 2]
                    );

                    calcPolyVertexTo.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 1]
                    );
                    calcPolyVertexTo.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 2]
                    );
                    calcPolyVertexTo.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 0]
                    );

                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );
                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );
                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );

                    useInLoopPolySurfaceIinfo.indexViewVolumeLineA = j;
                    useInLoopPolySurfaceIinfo.indexWithinRangeAryNum = i;
                    useInLoopPolySurfaceIinfo.polySum = polySum;
                    useInLoopPolySurfaceIinfo.polyTrueI = true;
                    useInLoopPolySurfaceIinfo.indexPolyI = k;
                    
                    polySurfaceIinfo.push_back(useInLoopPolySurfaceIinfo);
                    
                    linePlaneIindex += 1;
                }
            }

            polySum +=  numPolyTrueIViewVolume[i].size();
        }

        for (int i = 0; i < withinRangeAryNum.size(); ++i)
        {
            for (int k = 0; k < indexInViewVolumeAllOutside[i].size(); ++k)
            {
                if (eq.existenceI[j][polySum + k] == I_TRUE)
                {
                    calcPolyVertex.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 0
                        ]
                    );
                    calcPolyVertex.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 1
                        ]
                    );
                    calcPolyVertex.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 2
                        ]
                    );

                    calcPolyVertexTo.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 1
                        ]
                    );
                    calcPolyVertexTo.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 2
                        ]
                    );
                    calcPolyVertexTo.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 0
                        ]
                    );

                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );
                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );
                    calcPolySurfaceIpoint.push_back
                    (
                        eq.linePlaneI[linePlaneIindex]
                    );

                    useInLoopPolySurfaceIinfo.indexViewVolumeLineA = j;
                    useInLoopPolySurfaceIinfo.indexWithinRangeAryNum = i;
                    useInLoopPolySurfaceIinfo.polySum = polySum;
                    useInLoopPolySurfaceIinfo.polyTrueI = false;
                    useInLoopPolySurfaceIinfo.indexPolyI = k;
                    polySurfaceIinfo.push_back(useInLoopPolySurfaceIinfo);

                    linePlaneIindex += 1;
                }
            }
            polySum +=  indexInViewVolumeAllOutside[i].size();
        }

        polySum = 0;
    }

    vec.minusVec3d(calcPolyVertexTo, calcPolySurfaceIpoint);
    std::vector<VECTOR3D> vecPolyVtoI = vec.resultVector3D;

    vec.minusVec3d(calcPolyVertex, calcPolyVertexTo);
    std::vector<VECTOR3D> vecPolyLine = vec.resultVector3D;

    vec.crossProduct(vecPolyVtoI, vecPolyLine);

    std::vector<VECTOR3D> calcDotA;
    std::vector<VECTOR3D> calcDotB;
    calcDotA.resize(vec.resultVector3D.size() / 3 * 2);
    calcDotB.resize(vec.resultVector3D.size() / 3 * 2);

    for (int i = 0; i < vec.resultVector3D.size() / 3; ++i)
    {
        calcDotA[i*2 + 0] = vec.resultVector3D[i*3 + 0];
        calcDotA[i*2 + 1] = vec.resultVector3D[i*3 + 0];

        calcDotB[i*2 + 0] = vec.resultVector3D[i*3 + 1];
        calcDotB[i*2 + 1] = vec.resultVector3D[i*3 + 2];
    }

    vec.dotProduct(calcDotA, calcDotB);

    // polySum = 0;
    // linePlaneIindex = 0;

    for (int i = 0; i < polySurfaceIinfo.size(); ++i)
    {
        if (vec.resultVector[i*2 + 0] > 0 && vec.resultVector[i*2 + 1] > 0)
        {
            if (polySurfaceIinfo[i].polyTrueI)
            {
                clippedPolyVertex[polySurfaceIinfo[i].indexWithinRangeAryNum]
                [
                    indexNumPolyFacing
                    [
                        polySurfaceIinfo[i].indexWithinRangeAryNum
                    ]
                    [
                        numPolyTrueIViewVolume
                        [polySurfaceIinfo[i].indexWithinRangeAryNum][polySurfaceIinfo[i].polyTrueI]
                    ]*2 + 1
                ].push_back(eq.linePlaneI[i]);
            }
            else
            {
                numPolyInViewVolume[polySurfaceIinfo[i].indexWithinRangeAryNum]
                [
                    indexNumPolyFacing[polySurfaceIinfo[i].indexWithinRangeAryNum]
                    [
                        numPolyAllVLINENotInViewVolume[polySurfaceIinfo[i].indexWithinRangeAryNum]
                        [
                            indexInViewVolumeAllOutside
                            [polySurfaceIinfo[i].indexWithinRangeAryNum][polySurfaceIinfo[i].indexPolyI]
                        ]
                    ]
                ] = numPolyExitsIViewVolume
                [
                    polySurfaceIinfo[i].indexWithinRangeAryNum
                ]
                [
                    numPolyAllVLINENotInViewVolume[polySurfaceIinfo[i].indexWithinRangeAryNum]
                    [
                        indexInViewVolumeAllOutside
                        [polySurfaceIinfo[i].indexWithinRangeAryNum][polySurfaceIinfo[i].indexPolyI]
                    ]
                ];

                clippedPolyVertex[polySurfaceIinfo[i].indexWithinRangeAryNum]
                [
                    indexNumPolyFacing[polySurfaceIinfo[i].indexWithinRangeAryNum]
                    [
                        numPolyAllVLINENotInViewVolume
                        [polySurfaceIinfo[polySurfaceIinfo[i].indexWithinRangeAryNum].indexWithinRangeAryNum]
                        [
                            indexInViewVolumeAllOutside
                            [polySurfaceIinfo[i].indexWithinRangeAryNum][polySurfaceIinfo[i].indexPolyI]
                        ]
                    ] * 2 + 1
                ].push_back(eq.linePlaneI[i]);
            }
        }
    }
    
    // for (int j = 0; j < viewVolumeLineA.size(); ++j)
    // {
    //     for (int i = 0; i < withinRangeAryNum.size(); ++i)
    //     {
    //         for (int k = 0; k < numPolyTrueIViewVolume[i].size(); ++k)
    //         {
    //             if (eq.existenceI[j][polySum + k] == I_TRUE)
    //             {
    //                 clippedPolyVertex[i]
    //                 [
    //                     indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*2 + 1
    //                 ].push_back(eq.linePlaneI[linePlaneIindex]);
    //                 linePlaneIindex += 1;
    //             }
    //         }

    //         polySum +=  numPolyTrueIViewVolume[i].size();
    //     }

    //     for (int i = 0; i < withinRangeAryNum.size(); ++i)
    //     {
    //         for (int k = 0; k < indexInViewVolumeAllOutside[i].size(); ++k)
    //         {
    //             if (eq.existenceI[j][polySum + k] == I_TRUE)
    //             {
    //                 numPolyInViewVolume[i]
    //                 [
    //                     indexNumPolyFacing[i][numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]]
    //                 ] = numPolyExitsIViewVolume[i][indexInViewVolumeAllOutside[i][k]];
    //                 clippedPolyVertex[i]
    //                 [
    //                     indexNumPolyFacing[i]
    //                     [numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]] * 2 + 1
    //                 ].push_back(eq.linePlaneI[linePlaneIindex]);
    //                 linePlaneIindex += 1;
    //             }
    //         }
    //         polySum +=  indexInViewVolumeAllOutside[i].size();
    //     }

    //     polySum = 0;
    // }
    
}
