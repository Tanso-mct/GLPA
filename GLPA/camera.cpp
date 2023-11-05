#define DEBUG_CAMERA_
#include "camera.h"

void Camera::initialize()
{
    clipPolyInfo.meshID.resize(NULL);
    clipPolyInfo.polyID.resize(NULL);
    clipPolyInfo.oneV.resize(NULL);
    clipPolyInfo.normal.resize(NULL);

    sourcePolyInfo.resize(NULL);
    calcPolyInfo.resize(NULL);
    searchPolyInfo.resize(NULL);
    renderSouce.resize(NULL);
}

void Camera::coordinateTransRange(std::vector<OBJ_FILE>* meshData)
{
    // origin and opposite point data
    std::vector<VECTOR3D> pointData;

    for (int i = 0; i < (*meshData).size(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            pointData.push_back((*meshData)[i].range.wVertex[j]);
        }
    }

    vec.posTrans(pointData, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);

    for (int i = 0; i < (*meshData).size(); ++i)
    {
        (*meshData)[i].range.origin = mtx.resultMatrices[i*8];
        (*meshData)[i].range.opposite = mtx.resultMatrices[i*8];
    }
    
    for (int i = 0; i < (*meshData).size(); ++i)
    {
        for (int j = 1; j < 8; ++j)
        {
            // Processing with respect to origin point
            if ((*meshData)[i].range.origin.x > mtx.resultMatrices[i*8 + j].x)
            {   
                (*meshData)[i].range.origin.x = mtx.resultMatrices[i*8 + j].x;
            }
            if ((*meshData)[i].range.origin.y > mtx.resultMatrices[i*8 + j].y)
            {
                (*meshData)[i].range.origin.y = mtx.resultMatrices[i*8 + j].y;
            }
            if ((*meshData)[i].range.origin.z < mtx.resultMatrices[i*8 + j].z)
            {
                (*meshData)[i].range.origin.z = mtx.resultMatrices[i*8 + j].z;
            }

            // Processing with respect to opposite point
            if ((*meshData)[i].range.opposite.x < mtx.resultMatrices[i*8 + j].x)
            {
                (*meshData)[i].range.opposite.x = mtx.resultMatrices[i*8 + j].x;
            }
            if ((*meshData)[i].range.opposite.y < mtx.resultMatrices[i*8 + j].y)
            {
                (*meshData)[i].range.opposite.y = mtx.resultMatrices[i*8 + j].y;
            }
            if ((*meshData)[i].range.opposite.z > mtx.resultMatrices[i*8 + j].z)
            {
                (*meshData)[i].range.opposite.z = mtx.resultMatrices[i*8 + j].z;
            }
        }
    }
    
}

void Camera::clipRange(std::vector<OBJ_FILE> meshData)
{
    std::vector<double> vertValue;
    std::vector<double> horizValue;

    vertValue.resize(meshData.size() * 4);
    horizValue.resize(meshData.size() * 4);

    for (int i = 0; i < meshData.size(); ++i)
    {
        // origin XZ
        vertValue[i*4 + 0] = meshData[i].range.opposite.z;
        horizValue[i*4 + 0] = meshData[i].range.origin.x;

        // origin YZ
        vertValue[i*4 + 1] = meshData[i].range.opposite.z;
        horizValue[i*4 + 1] = meshData[i].range.origin.y;

        // opposite XZ
        vertValue[i*4 + 2] = meshData[i].range.opposite.z;
        horizValue[i*4 + 2] = meshData[i].range.opposite.x;

        // opposite YZ
        vertValue[i*4 + 3] = meshData[i].range.opposite.z;
        horizValue[i*4 + 3] = meshData[i].range.opposite.y;
    }

    tri.get2dVecAngle(vertValue, horizValue);
    for (int i = 0; i < meshData.size(); ++i)
    {
        if (viewVolume.clip(meshData, tri.resultDegree, viewAngle, i) != -1)
        {
            for (int j = 0; j < meshData[i].poly.v.size(); ++j)
            {
                clipPolyInfo.meshID.push_back(i);
                clipPolyInfo.polyID.push_back(j);
                clipPolyInfo.oneV.push_back(meshData[i].v.world[meshData[i].poly.v[j].num1]);
                clipPolyInfo.normal.push_back(meshData[i].v.normal[meshData[i].poly.normal[j].num1]);
            }
        }
    }
}

std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> Camera::polyBilateralJudge(std::vector<OBJ_FILE> meshData)
{
    // Camera coordinate transformation of the vertices of a face
    vec.posTrans(clipPolyInfo.oneV, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);
    clipPolyInfo.oneV = mtx.resultMatrices;

    // Camera coordinate transformation of the normal vector of a surface
    mtx.rotTrans(clipPolyInfo.normal, rotAngle);
    clipPolyInfo.normal = mtx.resultMatrices;

    vec.dotProduct(clipPolyInfo.normal, clipPolyInfo.oneV);

    // Creation of source polygon information structure
    std::vector<VECTOR3D> rtCalcPolyV;
    std::vector<VECTOR3D> rtCalcPolyNormal;
    POLYINFO pushPolyInfo;
    pushPolyInfo.lineStartPoint.resize(3);
    pushPolyInfo.lineEndPoint.resize(3);
    pushPolyInfo.lineVec.resize(3);
    for (int i = 0; i < clipPolyInfo.meshID.size(); ++i)
    {
        if (vec.resultVector[i] < 0)
        {
            pushPolyInfo.meshID = clipPolyInfo.meshID[i];
            pushPolyInfo.polyID = clipPolyInfo.polyID[i];

            sourcePolyInfo.push_back(pushPolyInfo);

            rtCalcPolyV.push_back
            (
                meshData[clipPolyInfo.meshID[i]].v.world
                [meshData[clipPolyInfo.meshID[i]].poly.v[clipPolyInfo.polyID[i]].num1]
            );

            rtCalcPolyV.push_back
            (
                meshData[clipPolyInfo.meshID[i]].v.world
                [meshData[clipPolyInfo.meshID[i]].poly.v[clipPolyInfo.polyID[i]].num2]
            );

            rtCalcPolyV.push_back
            (
                meshData[clipPolyInfo.meshID[i]].v.world
                [meshData[clipPolyInfo.meshID[i]].poly.v[clipPolyInfo.polyID[i]].num3]
            );
            
            rtCalcPolyNormal.push_back
            (
                meshData[clipPolyInfo.meshID[i]].v.normal
                [meshData[clipPolyInfo.meshID[i]].poly.normal[clipPolyInfo.polyID[i]].num1]
            );
        }
    }

    return std::make_tuple(rtCalcPolyV, rtCalcPolyNormal);
}

void Camera::coordinateTrans(std::vector<OBJ_FILE> meshData)
{
    // Stores all vertices of surface polygons
    std::tuple<std::vector<VECTOR3D>, std::vector<VECTOR3D>> calcPolyData = polyBilateralJudge(meshData);

    std::vector<VECTOR3D> calcPolyV = std::get<0>(calcPolyData);
    std::vector<VECTOR3D> calcPolyNormal = std::get<1>(calcPolyData);

    // Camera coordinate transformation of vertex data
    vec.posTrans(calcPolyV, wPos);
    mtx.rotTrans(vec.resultVector3D, rotAngle);
    calcPolyV = mtx.resultMatrices;

    mtx.rotTrans(calcPolyNormal, rotAngle);
    calcPolyNormal = mtx.resultMatrices;

    // Input to source polygon information structure
    for (int i = 0; i < sourcePolyInfo.size(); ++i)
    {
        sourcePolyInfo[i].lineStartPoint[VX]= calcPolyV[i*POLYV + VX];
        sourcePolyInfo[i].lineStartPoint[VY]= calcPolyV[i*POLYV + VY];
        sourcePolyInfo[i].lineStartPoint[VZ]= calcPolyV[i*POLYV + VZ];

        sourcePolyInfo[i].lineEndPoint[VX]= calcPolyV[i*POLYV + VY];
        sourcePolyInfo[i].lineEndPoint[VY]= calcPolyV[i*POLYV + VZ];
        sourcePolyInfo[i].lineEndPoint[VZ]= calcPolyV[i*POLYV + VX];

        sourcePolyInfo[i].polyNormal = calcPolyNormal[i];
    }
}

bool Camera::clipVerticesViewVolume()
{
    std::vector<double> calcVertValue;
    std::vector<double> calcHorizValue;

    calcVertValue.resize(sourcePolyInfo.size() * 3 * 2)
}

std::vector<bool> Camera::vertexInViewVolume(std::vector<VECTOR3D> v)
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

void Camera::polyInViewVolumeJudge(std::vector<OBJ_FILE> meshData)
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
