#define DEBUG_CAMERA_
#include "camera.h"

void CAMERA::initialize()
{
    wPos = {0, -350, 0};
    rotAngle = {0, 0, 0};

    nearZ = 1;
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
    viewVolumeFaceVertex[SURFACE_TOP] = viewPoint[0];
    viewVolumeFaceVertex[SURFACE_FRONT] = viewPoint[0];
    viewVolumeFaceVertex[SURFACE_RIGHT] = viewPoint[6];
    viewVolumeFaceVertex[SURFACE_LEFT] = viewPoint[0];
    viewVolumeFaceVertex[SURFACE_BACK] = viewPoint[6];
    viewVolumeFaceVertex[SURFACE_BOTTOM] = viewPoint[6];


    std::vector<VECTOR3D> calcVecA;
    calcVecA.push_back(viewPoint[1]);
    calcVecA.push_back(viewPoint[4]);
    calcVecA.push_back(viewPoint[1]);
    calcVecA.push_back(viewPoint[3]);
    calcVecA.push_back(viewPoint[2]);
    calcVecA.push_back(viewPoint[5]);
    calcVecA.push_back(viewPoint[3]);
    calcVecA.push_back(viewPoint[4]);
    calcVecA.push_back(viewPoint[5]);
    calcVecA.push_back(viewPoint[7]);
    calcVecA.push_back(viewPoint[2]);
    calcVecA.push_back(viewPoint[7]);

    std::vector<VECTOR3D> calcVecB;
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[6]);
    calcVecB.push_back(viewPoint[6]);
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[0]);
    calcVecB.push_back(viewPoint[6]);
    calcVecB.push_back(viewPoint[6]);
    calcVecB.push_back(viewPoint[6]);
    calcVecB.push_back(viewPoint[6]);

    vec.minusVec3d(calcVecA, calcVecB);

    std::vector<VECTOR3D> lineA;
    std::vector<VECTOR3D> lineB;
    lineA.resize(6);
    lineB.resize(6);

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            lineA[i] = vec.resultVector3D[i*2 + 0];
            lineB[i] = vec.resultVector3D[i*2 + 1];
        }
    }

    vec.crossProduct(lineA, lineB);

    viewVolumeFaceNormal.resize(6);
    viewVolumeFaceNormal = vec.resultVector3D;

    VECTOR3D storeNormal;
    double inSqrt;
    for (int i = 0; i < 6; ++i)
    {
        storeNormal = viewVolumeFaceNormal[i];
        inSqrt = pow(storeNormal.x, 2) + pow(storeNormal.y, 2) + pow(storeNormal.z, 2);

        viewVolumeFaceNormal[i].x = viewVolumeFaceNormal[i].x / sqrt(inSqrt);
        viewVolumeFaceNormal[i].y = viewVolumeFaceNormal[i].y / sqrt(inSqrt);
        viewVolumeFaceNormal[i].z = viewVolumeFaceNormal[i].z / sqrt(inSqrt);
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
                objData[i].range.origin.z < 
                ((viewPointXZ[VP3].z - viewPointXZ[VP4].z) / (viewPointXZ[VP3].x - viewPointXZ[VP4].x))
                * (objData[i].range.origin.x - viewPointXZ[VP4].x) 
                + viewPointXZ[VP4].z &&

                // OPPOSITE
                objData[i].range.opposite.z < 
                ((viewPointXZ[VP2].z - viewPointXZ[VP1].z) / (viewPointXZ[VP2].x - viewPointXZ[VP1].x)) 
                * (objData[i].range.opposite.x - viewPointXZ[VP1].x) 
                + viewPointXZ[VP1].z
            )
            {
                if
                (
                    // Y-axis direction determination
                    // ORIGIN
                    objData[i].range.origin.y < 
                    ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                    * (objData[i].range.origin.z - viewPointYZ[VP1].z) 
                    + viewPointYZ[VP1].y &&

                    // OPPOSIT
                    objData[i].range.opposite.y > 
                    ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                    * (objData[i].range.opposite.z - viewPointYZ[VP4].z) 
                    + viewPointYZ[VP4].y
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

std::vector<std::vector<int>> CAMERA::clippingRange(std::vector<std::vector<RANGE_CUBE_POLY>> rangePoly, int processObjectAmout)
{
    std::vector<std::vector<int>> indexInViewVolumePoly;
    indexInViewVolumePoly.resize(processObjectAmout);
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
                    rangePoly[i][j].origin.z < 
                    ((viewPointXZ[VP3].z - viewPointXZ[VP4].z) / (viewPointXZ[VP3].x - viewPointXZ[VP4].x))
                    * (rangePoly[i][j].origin.x - viewPointXZ[VP4].x) 
                    + viewPointXZ[VP4].z &&

                    // OPPOSITE
                    rangePoly[i][j].opposite.z < 
                    ((viewPointXZ[VP2].z - viewPointXZ[VP1].z) / (viewPointXZ[VP2].x - viewPointXZ[VP1].x)) 
                    * (rangePoly[i][j].opposite.x - viewPointXZ[VP1].x) 
                    + viewPointXZ[VP1].z
                )
                {
                    if
                    (
                        // Y-axis direction determination
                        // ORIGIN
                        rangePoly[i][j].origin.y < 
                        ((viewPointYZ[VP2].y - viewPointYZ[VP1].y) / (viewPointYZ[VP2].z - viewPointYZ[VP1].z)) 
                        * (rangePoly[i][j].origin.z - viewPointYZ[VP1].z) 
                        + viewPointYZ[VP1].y &&

                        // OPPOSIT
                        rangePoly[i][j].opposite.y > 
                        ((viewPointYZ[VP3].y - viewPointYZ[VP4].y) / (viewPointYZ[VP3].z - viewPointYZ[VP4].z)) 
                        * (rangePoly[i][j].opposite.z - viewPointYZ[VP4].z) 
                        + viewPointYZ[VP4].y
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
    for (int i = 0; i < withinRangeAryNum.size(); ++i)
    {
        numPolyInViewVolume[i].resize(numPolyFacing[i].size(), -1);
        clippedPolyVertex[i].resize(numPolyFacing[i].size()*2);
        for (int j = 0; j < numPolyFacing[i].size(); ++j)
        {
            if (vertexInViewVolume(polyVertex[polySum + j*3 + 0]))
            {
                clippedPolyVertex[i][j*2].push_back(polyVertex[polySum + j*3 + 0]);
                inVolumeAmoutV += 1;
            }
            if (vertexInViewVolume(polyVertex[polySum + j*3 + 1]))
            {
                clippedPolyVertex[i][j*2].push_back(polyVertex[polySum + j*3 + 1]);
                inVolumeAmoutV += 1;
            }
            if (vertexInViewVolume(polyVertex[polySum + j*3 + 2]))
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

    numPolyAllVLINENotInViewVolume.resize(0);
    numPolyAllVLINENotInViewVolume.resize(withinRangeAryNum.size());

    polySum = 0;
    int linePlaneIindex = 0;
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
                        numPolyInViewVolume[i][indexNumPolyFacing[i][j]] = numPolyExitsIViewVolume[i][j];
                        clippedPolyVertex[i][indexNumPolyFacing[i][j]*2 + 1].push_back(eq.linePlaneI[linePlaneIindex]);
                        linePlaneIindex += 1;
                        findTrueI = true;
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
            viewVolumeLineB.push_back(viewPoint[0]);
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
    std::vector<VECTOR3D> calcPolyVertexFrom;
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

                    calcPolyVertexFrom.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 2]
                    );
                    calcPolyVertexFrom.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 0]
                    );
                    calcPolyVertexFrom.push_back
                    (
                        polyVertex[indexNumPolyFacing[i][numPolyTrueIViewVolume[i][k]]*3 + 1]
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
                    // numPolyInViewVolume[i]
                    // [
                    //     indexNumPolyFacing[i][numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]]
                    // ] = numPolyExitsIViewVolume[i][indexInViewVolumeAllOutside[i][k]];
                    // clippedPolyVertex[i]
                    // [
                    //     indexNumPolyFacing[i]
                    //     [numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]] * 2 + 1
                    // ].push_back(eq.linePlaneI[linePlaneIindex]);

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

                    calcPolyVertexFrom.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 2
                        ]
                    );
                    calcPolyVertexFrom.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 0
                        ]
                    );
                    calcPolyVertexFrom.push_back
                    (
                        polyVertex
                        [
                            indexNumPolyFacing[i]
                            [
                                numPolyAllVLINENotInViewVolume[i][indexInViewVolumeAllOutside[i][k]]
                            ]*3 + 1
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

    vec.minusVec3d(calcPolyVertex, calcPolySurfaceIpoint);
    std::vector<VECTOR3D> vecPolyVtoI = vec.resultVector3D;

    vec.minusVec3d(calcPolyVertex, calcPolyVertexFrom);
    std::vector<VECTOR3D> vecPolyLine = vec.resultVector3D;

    vec.crossProduct(vecPolyLine, vecPolyVtoI);

    std::vector<VECTOR3D> calcDotA;
    std::vector<VECTOR3D> calcDotB;
    calcDotA.resize(vec.resultVector3D.size() / 3 * 2);
    calcDotB.resize(vec.resultVector3D.size() / 3 * 2);

    for (int i = 0; i < vec.resultVector3D.size() / 3; ++i)
    {
        calcDotA[i + 0] = vec.resultVector3D[i + 0];
        calcDotA[i + 1] = vec.resultVector3D[i + 0];

        calcDotB[i + 0] = vec.resultVector3D[i + 1];
        calcDotB[i + 1] = vec.resultVector3D[i + 2];
    }

    vec.crossProduct(calcDotA, calcDotB);

    // polySum = 0;
    // linePlaneIindex = 0;

    for (int i = 0; i < polySurfaceIinfo.size(); ++i)
    {
        if (vec.resultVector[i*2 + 0] > 0 || vec.resultVector[i*2 + 1] > 0)
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
                    indexInViewVolumeAllOutside
                    [polySurfaceIinfo[i].indexWithinRangeAryNum][polySurfaceIinfo[i].indexPolyI]
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
