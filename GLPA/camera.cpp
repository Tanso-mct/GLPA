#define DEBUG_CAMERA_
#include "camera.h"

void CAMERA::initialize()
{
    wPos = {0, -300, 0};
    rotAngle = {0, 0, 0};

    nearZ = 0.01;
    farZ = 10000;
    viewAngle = 80;
    aspectRatio = {16, 9};
}

void CAMERA::defViewVolume()
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

    viewPointXZ[VP1].z = -nearZ;
    viewPointXZ[VP2].z = -farZ;
    viewPointXZ[VP3].z = -farZ;
    viewPointXZ[VP4].z = -nearZ;

    // Define coordinates of view area vertices on yz axis
    viewPointYZ[VP1].y = nearScreenSize.height / 2;
    viewPointYZ[VP2].y = farScreenSize.height / 2;
    viewPointYZ[VP3].y = -farScreenSize.height / 2;
    viewPointYZ[VP4].y = -nearScreenSize.height / 2;

    viewPointYZ[VP1].z = -nearZ;
    viewPointYZ[VP2].z = -farZ;
    viewPointYZ[VP3].z = -farZ;
    viewPointYZ[VP4].z = -nearZ;

    viewPoint.resize(8);
    // 0
    viewPoint[0].x = viewPointXZ[VP1].x;
    viewPoint[0].y = viewPointYZ[VP1].y;
    viewPoint[0].z = -nearZ;

    // 1
    viewPoint[1].x = viewPointXZ[VP4].x;
    viewPoint[1].y = viewPointYZ[VP1].y;
    viewPoint[1].z = -nearZ;

    // 2
    viewPoint[2].x = viewPointXZ[VP4].x;
    viewPoint[2].y = viewPointYZ[VP4].y;
    viewPoint[2].z = -nearZ;

    // 3
    viewPoint[3].x = viewPointXZ[VP1].x;
    viewPoint[3].y = viewPointYZ[VP4].y;
    viewPoint[3].z = -nearZ;

    // 4
    viewPoint[4].x = viewPointXZ[VP2].x;
    viewPoint[4].y = viewPointYZ[VP2].y;
    viewPoint[4].z = -farZ;

    // 5
    viewPoint[5].x = viewPointXZ[VP3].x;
    viewPoint[5].y = viewPointYZ[VP2].y;
    viewPoint[5].z = -farZ;

    // 6
    viewPoint[6].x = viewPointXZ[VP3].x;
    viewPoint[6].y = viewPointYZ[VP3].y;
    viewPoint[6].z = -farZ;

    // 7
    viewPoint[7].x = viewPointXZ[VP2].x;
    viewPoint[7].y = viewPointYZ[VP3].y;
    viewPoint[7].z = -farZ;

    // Assign a point on the surface
    viewVolumeFaceVertex.resize(6);
    viewVolumeFaceVertex[SURFACE_TOP] = viewPoint[5];
    viewVolumeFaceVertex[SURFACE_FRONT] = viewPoint[3];
    viewVolumeFaceVertex[SURFACE_RIGHT] = viewPoint[5];
    viewVolumeFaceVertex[SURFACE_LEFT] = viewPoint[3];
    viewVolumeFaceVertex[SURFACE_BACK] = viewPoint[5];
    viewVolumeFaceVertex[SURFACE_BOTTOM] = viewPoint[3];

    std::vector<VECTOR3D> calcViewPoint;
    calcViewPoint.resize(6);
    calcViewPoint[SURFACE_TOP] = viewPoint[0];
    calcViewPoint[SURFACE_FRONT] = viewPoint[0];
    calcViewPoint[SURFACE_RIGHT] = viewPoint[6];
    calcViewPoint[SURFACE_LEFT] = viewPoint[0];
    calcViewPoint[SURFACE_BACK] = viewPoint[6];
    calcViewPoint[SURFACE_BOTTOM] = viewPoint[6];

    vec.crossProduct(viewVolumeFaceVertex, calcViewPoint);

    viewVolumeFaceNormal.resize(6);
    viewVolumeFaceNormal = vec.resultVector3D;
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
    withinRangeAryNum.resize(0);
    for (int i = 0; i < objData.size(); ++i)
    {
        // Z-axis direction determination
        if 
        (
            objData[i].range.origin.z > viewPointXZ[VP2].z ||
            objData[i].range.opposite.z < viewPointXZ[VP1].z
        )
        {
            // X-axis direction determination
            if 
            (
                // ORIGIN
                (objData[i].range.origin.x < viewPointXZ[VP3].x &&
                objData[i].range.origin.x < 
                ((viewPointXZ[VP3].x - viewPointXZ[VP4].x) / (viewPointXZ[VP3].z - viewPointXZ[VP4].z))
                * (objData[i].range.origin.z - viewPointXZ[VP4].z) 
                + viewPointXZ[VP4].x) ||

                // OPPOSITE
                (objData[i].range.opposite.x > viewPointXZ[VP2].x &&
                objData[i].range.opposite.x > 
                ((viewPointXZ[VP2].x - viewPointXZ[VP1].x) / (viewPointXZ[VP2].z - viewPointXZ[VP1].z)) 
                * (objData[i].range.opposite.z - viewPointXZ[VP1].z) 
                + viewPointXZ[VP1].x)
            )
            {
                if
                (
                    // Y-axis direction determination
                    // ORIGIN
                    (objData[i].range.origin.y < viewPointYZ[VP2].y &&
                    objData[i].range.origin.y < 
                    ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                    * (objData[i].range.origin.z - viewPointYZ[VP1].z) 
                    + viewPointYZ[VP1].y) ||

                    // OPPOSIT
                    (objData[i].range.opposite.y > viewPointYZ[VP3].y &&
                    objData[i].range.opposite.y > 
                    ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                    * (objData[i].range.opposite.z - viewPointYZ[VP4].z) 
                    + viewPointYZ[VP4].y)
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
                numPolyFacing[i].n.push_back(j);
            }
        }
        sumFaceAmout += faceAmout[i];
    }
}

void CAMERA::coordinateTransV(std::vector<OBJ_FILE> objData)
{
    // Stores all vertices of surface polygons
    polyVertex.resize(0);
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        for (int j = 0; j < numPolyFacing[i].n.size(); ++j)
        {
            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i].n[j]].num1
                ]
            );

            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i].n[j]].num2
                ]
            );

            polyVertex.push_back
            (
                objData[i].v.world
                [
                    objData[i].poly.v[numPolyFacing[i].n[j]].num3
                ]   
            );
        }
    }

    // Camera coordinate transformation of vertex data
    vec.posTrans(polyVertex, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);
    polyVertex = mtx.resultMatrices;
}

bool CAMERA::confirmI
(
    double exitsI, 
    double leftLessThan1Data, double rightLessThan1Data,
    double leftGreaterThan1Data, double rightGreaterThan1Data, 
    double leftLessThan2Data, double rightLessThan2Data,
    double leftGreaterThan2Data, double rightGreaterThan2Data,
    VECTOR3D linePlaneI, VECTOR3D lineVA, VECTOR3D lineVB,
    int withInRangeAryNumLoopN, int numPolyfacingLoopN
)
{
    if (!std::isinf(exitsI))
    {
        if 
        (
            leftLessThan1Data < rightLessThan1Data &&
            leftGreaterThan1Data > rightGreaterThan1Data &&
            leftLessThan2Data < rightLessThan2Data &&
            leftGreaterThan2Data > rightGreaterThan2Data &&
            sqrt(pow(lineVB.x - lineVA.x, 2) + pow(lineVB.y - lineVA.y, 2) + pow(lineVB.z - lineVA.z, 2))
            > sqrt(pow(linePlaneI.x - lineVA.x, 2) + pow(linePlaneI.y - lineVA.y, 2) + pow(linePlaneI.z - lineVA.z, 2))
        )
        {
            numPolyInViewVolume[withInRangeAryNumLoopN].n.push_back
            (numPolyExitsIViewVolume[withInRangeAryNumLoopN].n[numPolyfacingLoopN]);
            return true;
        }
        return false;
    }
}

bool CAMERA::vertexInViewVolume(VECTOR3D v)
{
    if 
    (
        v.z > viewPointXZ[VP2].z && v.z < viewPointXZ[VP1].z
    )
    {
        // X-axis direction determination
        if 
        (
            // ORIGIN
            v.x < viewPointXZ[VP3].x &&
            v.x < 
            ((viewPointXZ[VP3].x - viewPointXZ[VP4].x) / (viewPointXZ[VP3].z - viewPointXZ[VP4].z))
            * (v.z - viewPointXZ[VP4].z) + viewPointXZ[VP4].x &&

            // OPPOSITE
            v.x > viewPointXZ[VP2].x &&
            v.x > 
            ((viewPointXZ[VP2].x - viewPointXZ[VP1].x) / (viewPointXZ[VP2].z - viewPointXZ[VP1].z)) 
            * (v.z - viewPointXZ[VP1].z) + viewPointXZ[VP1].x
        )
        {
            if
            (
                // Y-axis direction determination
                // ORIGIN
                v.y < viewPointYZ[VP2].y &&
                v.y < 
                ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                * (v.z - viewPointYZ[VP1].z) + viewPointYZ[VP1].y &&

                // OPPOSIT
                v.y > viewPointYZ[VP3].y &&
                v.y > 
                ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                * (v.z - viewPointYZ[VP4].z) + viewPointYZ[VP4].y
            )
            {
                return true;
            }
        }
    }
    return false;
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
    clippingPolyVertex.resize(0);
    clippingPolyVertex.resize(withinRangeAryNum.size());
    
    // If one of the vertices is in the view volume, 
    // it is excluded from the intersection determination as a drawing target.
    int aryNum = 0;
    int inVolumeAmoutV = 0;
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        clippingPolyVertex[i].n.resize(numPolyFacing[i].n.size());
        for (int j = 0; j < numPolyFacing[i].n.size(); ++j)
        {
            if (vertexInViewVolume(polyVertex[aryNum + j*3 + 0]))
            {
                clippingPolyVertex[i].n[j].push_back(polyVertex[aryNum + j*3 + 0]);
                inVolumeAmoutV += 1;
            }
            if (vertexInViewVolume(polyVertex[aryNum + j*3 + 1]))
            {
                clippingPolyVertex[i].n[j].push_back(polyVertex[aryNum + j*3 + 1]);
                inVolumeAmoutV += 1;
            }
            if (vertexInViewVolume(polyVertex[aryNum + j*3 + 2]))
            {
                clippingPolyVertex[i].n[j].push_back(polyVertex[aryNum + j*3 + 2]);
                inVolumeAmoutV += 1;
            }

            if (inVolumeAmoutV == 3)
            {
                numPolyInViewVolume[i].n.push_back(numPolyFacing[i].n[j]);
            }
            else
            {
                numPolyExitsIViewVolume[i].n.push_back(numPolyFacing[i].n[j]);
                // Input vertex A
                polyLineVA.push_back
                (
                    polyVertex[aryNum + j*3 + 0]
                );

                polyLineVA.push_back
                (
                    polyVertex[aryNum + j*3 + 1]
                );

                polyLineVA.push_back
                (
                    polyVertex[aryNum + j*3 + 2]
                );

                // Input vertex B
                polyLineVB.push_back
                (
                    polyVertex[aryNum + j*3 + 1]
                );

                polyLineVB.push_back
                (
                    polyVertex[aryNum + j*3 + 2]
                );

                polyLineVB.push_back
                (
                    polyVertex[aryNum + j*3 + 0]
                );
            }

            inVolumeAmoutV = 0;
        }
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

    numPolyAllVINotInViewVolume.resize(withinRangeAryNum.size());

    aryNum = 0;
    int indexVertex = 0;
    bool findTrueI = false;

    for (int k = 0; k < withinRangeAryNum.size(); ++k)
    {
        for (int j = 0; j < numPolyExitsIViewVolume[k].n.size(); ++j)
        {
            for (int i = 0; i < 3; ++i)
            {
                int l;
                if (i == 0)
                {
                    l = 1;
                } else if (i == 1)
                {
                    l = 2;
                } else if (i == 2)
                {
                    l = 0;
                }
                // Judgment by XZ axis
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].x, 
                        ((viewPointXZ[VP3].x - viewPointXZ[VP4].x) / (viewPointXZ[VP3].z - viewPointXZ[VP4].z))
                        * (eq.linePlaneI[aryNum].z - viewPointXZ[VP4].z) 
                        + viewPointXZ[VP4].x,

                        eq.linePlaneI[aryNum].x,
                        ((viewPointXZ[VP2].x - viewPointXZ[VP1].x) / (viewPointXZ[VP2].z - viewPointXZ[VP1].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointXZ[VP1].z) 
                        + viewPointXZ[VP1].x,

                        eq.linePlaneI[aryNum].z, viewPointXZ[VP1].z,
                        eq.linePlaneI[aryNum].z, viewPointXZ[VP2].z,

                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 6;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }

                // Judgment by XY axis
                aryNum += 1;
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].x, viewPoint[1].x,
                        eq.linePlaneI[aryNum].x, viewPoint[0].x,
                        eq.linePlaneI[aryNum].y, viewPoint[0].y,
                        eq.linePlaneI[aryNum].y, viewPoint[3].y,
                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 5;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }

                // Judgment by YZ axis
                aryNum += 1;
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].y,
                        ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointYZ[VP1].z) 
                        + viewPointYZ[VP1].y,

                        eq.linePlaneI[aryNum].y,
                        ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointYZ[VP4].z) 
                        + viewPointYZ[VP4].y,

                        eq.linePlaneI[aryNum].z, viewPointXZ[VP1].z,
                        eq.linePlaneI[aryNum].z, viewPointXZ[VP2].z,

                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 4;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }

                // Judgment by YZ axis
                aryNum += 1;
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].y,
                        ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointYZ[VP1].z) 
                        + viewPointYZ[VP1].y,

                        eq.linePlaneI[aryNum].y,
                        ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointYZ[VP4].z) 
                        + viewPointYZ[VP4].y,

                        eq.linePlaneI[aryNum].z, viewPointXZ[VP1].z,
                        eq.linePlaneI[aryNum].z, viewPointXZ[VP2].z,

                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 3;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }

                // Judgment by XY axis
                aryNum += 1;
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].x, viewPoint[5].x,
                        eq.linePlaneI[aryNum].x, viewPoint[4].x,
                        eq.linePlaneI[aryNum].y, viewPoint[4].y,
                        eq.linePlaneI[aryNum].y, viewPoint[7].y,
                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 2;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }

                // Judgment by XZ axis
                aryNum += 1;
                if(
                    confirmI(
                        eq.linePlaneI[aryNum].x,

                        eq.linePlaneI[aryNum].x, 
                        ((viewPointXZ[VP3].x - viewPointXZ[VP4].x) / (viewPointXZ[VP3].z - viewPointXZ[VP4].z))
                        * (eq.linePlaneI[aryNum].z - viewPointXZ[VP4].z) 
                        + viewPointXZ[VP4].x,

                        eq.linePlaneI[aryNum].x,
                        ((viewPointXZ[VP2].x - viewPointXZ[VP1].x) / (viewPointXZ[VP2].z - viewPointXZ[VP1].z)) 
                        * (eq.linePlaneI[aryNum].z - viewPointXZ[VP1].z) 
                        + viewPointXZ[VP1].x,

                        eq.linePlaneI[aryNum].z, viewPointXZ[VP1].z,
                        eq.linePlaneI[aryNum].z, viewPointXZ[VP2].z,

                        eq.linePlaneI[aryNum], polyVertex[indexVertex + j*VECTOR3 + i], polyVertex[indexVertex + j*VECTOR3 + l],
                        k, j
                    )
                )
                {
                    aryNum += 1;
                    indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
                    findTrueI = true;
                    break;
                }
                aryNum += 1;
            }

            // Stores the number of polygons for which all intersections did not meet the condition
            if (!findTrueI)
            {
                numPolyAllVINotInViewVolume[k].n.push_back(numPolyExitsIViewVolume[k].n[j]);
            }
            
            indexVertex += numPolyExitsIViewVolume[k].n.size() * VECTOR3;
            findTrueI = false;
            
        }
    }
    
}

