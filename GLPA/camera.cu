#include "camera.cuh"


void Camera::load(
    std::wstring argName, 
    Vec3d argWPos, 
    Vec3d argRotAngle, 
    double argNearZ, 
    double argFarZ, 
    double argViewAngle, 
    Vec2d argAspectRatio,
    Vec2d argScPixelSize
){
    name = argName;
    wPos = argWPos;
    rotAngle = argRotAngle;
    nearZ = argNearZ;
    farZ = argFarZ;
    viewAngle = argViewAngle;
    aspectRatio = argAspectRatio;
    scPixelSize = argScPixelSize;

    reload = true;
}


void Camera::defineViewVolume(){
    // If no reloading has been done and no changes have been made to the definition, no processing is done.
    if (!reload){
        return;
    }

    // Get screen size
    nearScrSize.x = fabs(tan(RAD(viewAngle / 2)) * -nearZ) * 2;
    nearScrSize.y = fabs(nearScrSize.x * aspectRatio.y / aspectRatio.x);

    farScrSize.x = fabs(tan(RAD(viewAngle / 2)) * -farZ) * 2;
    farScrSize.y = fabs(farScrSize.x * aspectRatio.y / aspectRatio.x);

    viewAngleCos.x = cos(RAD(viewAngle / 2));
    viewAngleCos.y = fabs(-nearZ / sqrt(-nearZ*-nearZ + (nearScrSize.y/2) * (nearScrSize.y/2)));

    // farScrSize.x = nearScrSize.x / 2 * -farZ / -nearZ;
    // farScrSize.y = farScrSize.x * aspectRatio.y / aspectRatio.x;

    // Defines the coordinates of the four vertices when the view volume is viewed from the positive y-axis direction.
    viewVolume.xzV[0].x = -nearScrSize.x / 2;
    viewVolume.xzV[1].x  = -farScrSize.x / 2;
    viewVolume.xzV[2].x  = farScrSize.x / 2;
    viewVolume.xzV[3].x  = nearScrSize.x / 2;

    viewVolume.xzV[0].z = -nearZ;
    viewVolume.xzV[1].z = -farZ;
    viewVolume.xzV[2].z = -farZ;
    viewVolume.xzV[3].z = -nearZ;


    // Defines the coordinates of the four vertices when the view volume is viewed from the positive X-axis direction.
    viewVolume.yzV[0].y = nearScrSize.y / 2;
    viewVolume.yzV[1].y = farScrSize.y / 2;
    viewVolume.yzV[2].y = -farScrSize.y / 2;
    viewVolume.yzV[3].y = -nearScrSize.y / 2;

    viewVolume.yzV[0].z = -nearZ;
    viewVolume.yzV[1].z = -farZ;
    viewVolume.yzV[2].z = -farZ;
    viewVolume.yzV[3].z = -nearZ;


    // Defines the coordinates of the vertices in the camera 3D coordinates of the view volume.
    viewVolume.v[RECT_FRONT_TOP_LEFT].x = viewVolume.xzV[0].x;
    viewVolume.v[RECT_FRONT_TOP_LEFT].y = viewVolume.yzV[0].y;
    viewVolume.v[RECT_FRONT_TOP_LEFT].z = -nearZ;

    viewVolume.v[RECT_FRONT_TOP_RIGHT].x = viewVolume.xzV[3].x;
    viewVolume.v[RECT_FRONT_TOP_RIGHT].y = viewVolume.yzV[0].y;
    viewVolume.v[RECT_FRONT_TOP_RIGHT].z = -nearZ;

    viewVolume.v[RECT_FRONT_BOTTOM_RIGHT].x = viewVolume.xzV[3].x;
    viewVolume.v[RECT_FRONT_BOTTOM_RIGHT].y = viewVolume.yzV[3].y;
    viewVolume.v[RECT_FRONT_BOTTOM_RIGHT].z = -nearZ;

    viewVolume.v[RECT_FRONT_BOTTOM_LEFT].x = viewVolume.xzV[0].x;
    viewVolume.v[RECT_FRONT_BOTTOM_LEFT].y = viewVolume.yzV[3].y;
    viewVolume.v[RECT_FRONT_BOTTOM_LEFT].z = -nearZ;

    viewVolume.v[RECT_BACK_TOP_LEFT].x = viewVolume.xzV[1].x;
    viewVolume.v[RECT_BACK_TOP_LEFT].y = viewVolume.yzV[1].y;
    viewVolume.v[RECT_BACK_TOP_LEFT].z = -farZ;

    viewVolume.v[RECT_BACK_TOP_RIGHT].x = viewVolume.xzV[2].x;
    viewVolume.v[RECT_BACK_TOP_RIGHT].y = viewVolume.yzV[1].y;
    viewVolume.v[RECT_BACK_TOP_RIGHT].z = -farZ;

    viewVolume.v[RECT_BACK_BOTTOM_RIGHT].x = viewVolume.xzV[2].x;
    viewVolume.v[RECT_BACK_BOTTOM_RIGHT].y = viewVolume.yzV[2].y;
    viewVolume.v[RECT_BACK_BOTTOM_RIGHT].z = -farZ;

    viewVolume.v[RECT_BACK_BOTTOM_LEFT].x = viewVolume.xzV[1].x;
    viewVolume.v[RECT_BACK_BOTTOM_LEFT].y = viewVolume.yzV[2].y;
    viewVolume.v[RECT_BACK_BOTTOM_LEFT].z = -farZ;


    // Stores the coordinates of the vertices that are the start and end points of each line segment of the view volume.
    viewVolume.lines[0].startV = viewVolume.v[RECT_L1_STARTV];
    viewVolume.lines[1].startV = viewVolume.v[RECT_L2_STARTV];
    viewVolume.lines[2].startV = viewVolume.v[RECT_L3_STARTV];
    viewVolume.lines[3].startV = viewVolume.v[RECT_L4_STARTV];
    viewVolume.lines[4].startV = viewVolume.v[RECT_L5_STARTV];
    viewVolume.lines[5].startV = viewVolume.v[RECT_L6_STARTV];
    viewVolume.lines[6].startV = viewVolume.v[RECT_L7_STARTV];
    viewVolume.lines[7].startV = viewVolume.v[RECT_L8_STARTV];
    viewVolume.lines[8].startV = viewVolume.v[RECT_L9_STARTV];
    viewVolume.lines[9].startV = viewVolume.v[RECT_L10_STARTV];
    viewVolume.lines[10].startV = viewVolume.v[RECT_L11_STARTV];
    viewVolume.lines[11].startV = viewVolume.v[RECT_L12_STARTV];

    viewVolume.lines[0].endV = viewVolume.v[RECT_L1_ENDV];
    viewVolume.lines[1].endV = viewVolume.v[RECT_L2_ENDV];
    viewVolume.lines[2].endV = viewVolume.v[RECT_L3_ENDV];
    viewVolume.lines[3].endV = viewVolume.v[RECT_L4_ENDV];
    viewVolume.lines[4].endV = viewVolume.v[RECT_L5_ENDV];
    viewVolume.lines[5].endV = viewVolume.v[RECT_L6_ENDV];
    viewVolume.lines[6].endV = viewVolume.v[RECT_L7_ENDV];
    viewVolume.lines[7].endV = viewVolume.v[RECT_L8_ENDV];
    viewVolume.lines[8].endV = viewVolume.v[RECT_L9_ENDV];
    viewVolume.lines[9].endV = viewVolume.v[RECT_L10_ENDV];
    viewVolume.lines[10].endV = viewVolume.v[RECT_L11_ENDV];
    viewVolume.lines[11].endV = viewVolume.v[RECT_L12_ENDV];

    // Obtain a 3D vector for each line segment.
    for (auto& i : viewVolume.lines){
        i.vec.x = i.endV.x - i.startV.x;
        i.vec.y = i.endV.y - i.startV.y;
        i.vec.z = i.endV.z - i.startV.z;
    }

    // Stores the coordinates of the starting vertex of the normal vector for each face of the view volume.
    viewVolume.face.v[SURFACE_TOP] = viewVolume.v[RECT_FRONT_TOP_LEFT];
    viewVolume.face.v[SURFACE_FRONT] = viewVolume.v[RECT_FRONT_TOP_LEFT];
    viewVolume.face.v[SURFACE_RIGHT] = viewVolume.v[RECT_BACK_BOTTOM_RIGHT];
    viewVolume.face.v[SURFACE_LEFT] = viewVolume.v[RECT_FRONT_TOP_LEFT];
    viewVolume.face.v[SURFACE_BACK] = viewVolume.v[RECT_BACK_BOTTOM_RIGHT];
    viewVolume.face.v[SURFACE_BOTTOM] = viewVolume.v[RECT_BACK_BOTTOM_RIGHT];


    // Get the normal vector of each face of the view volume.
    std::vector<Vec3d> calcVA(6);
    std::vector<Vec3d> calcVB(6);

    calcVA[SURFACE_TOP] = viewVolume.lines[0].vec;
    calcVB[SURFACE_TOP] = viewVolume.lines[4].vec;

    calcVA[SURFACE_FRONT] = viewVolume.lines[0].vec;
    calcVB[SURFACE_FRONT] = viewVolume.lines[1].vec;

    calcVA[SURFACE_RIGHT] = viewVolume.lines[1].vec;
    calcVB[SURFACE_RIGHT] = viewVolume.lines[5].vec;

    calcVA[SURFACE_LEFT] = viewVolume.lines[3].vec;
    calcVB[SURFACE_LEFT] = viewVolume.lines[4].vec;

    calcVA[SURFACE_BACK] = viewVolume.lines[8].vec;
    calcVB[SURFACE_BACK] = viewVolume.lines[9].vec;

    calcVA[SURFACE_BOTTOM] = viewVolume.lines[2].vec;
    calcVB[SURFACE_BOTTOM] = viewVolume.lines[6].vec;

    for (int i = 0; i < 6; i++){
        viewVolume.face.normal[i].x = calcVA[i].y * calcVB[i].z - calcVA[i].z * calcVB[i].y;
        viewVolume.face.normal[i].y = calcVA[i].z * calcVB[i].x - calcVA[i].x * calcVB[i].z;
        viewVolume.face.normal[i].z = calcVA[i].x * calcVB[i].y - calcVA[i].y * calcVB[i].x;
    }

    reload = false;

}


void Camera::objCulling(std::unordered_map<std::wstring, Object> objects){
    std::vector<Vec3d> rangeVs(objects.size() * 8);

    int iN1 = 0;
    for (auto obj : objects){
        for (int i = 0; i < 8; i++){
            rangeVs[iN1*8 + i] = obj.second.range.wVertex[i];
        }
        iN1 += 1;
    }

    rangeVs = mt.transRotConvert(wPos, rotAngle, rangeVs);

    std::vector<std::wstring> objOrder;
    std::vector<Vec3d> oppositeSideXzVs;
    std::vector<Vec3d> oppositeSideYzVs;
    std::vector<Vec3d> oppositeSideVs;
    std::vector<double> orizinZ;
    std::vector<double> oppositeZ;

    std::vector<Vec3d> rectVs(2);
    bool status = false;
    Vec3d pushVec1;

    int iN2 = 0;

    for (auto obj : objects){
        objOrder.push_back(obj.first);
        
        status = false;
        for (int i = 0; i < 8; i++){
            if (status){
                if (rangeVs[iN2*8 + i].x < rectVs[0].x){
                    rectVs[0].x = rangeVs[iN2*8 + i].x;
                }
                if (rangeVs[iN2*8 + i].y < rectVs[0].y){
                    rectVs[0].y = rangeVs[iN2*8 + i].y;
                }
                if (rangeVs[iN2*8 + i].z > rectVs[0].z){
                    rectVs[0].z = rangeVs[iN2*8 + i].z;
                }

                // Processing with respect to opposite point
                if (rangeVs[iN2*8 + i].x > rectVs[1].x){
                    rectVs[1].x = rangeVs[iN2*8 + i].x;
                }
                if (rangeVs[iN2*8 + i].y > rectVs[1].y){
                    rectVs[1].y = rangeVs[iN2*8 + i].y;
                }
                if (rangeVs[iN2*8 + i].z < rectVs[1].z){
                    rectVs[1].z = rangeVs[iN2*8 + i].z;
                }
            }
            else{
                rectVs[0].x = rangeVs[iN2*8 + i].x;
                rectVs[0].y = rangeVs[iN2*8 + i].y;
                rectVs[0].z = rangeVs[iN2*8 + i].z;

                rectVs[1].x = rangeVs[iN2*8 + i].x;
                rectVs[1].y = rangeVs[iN2*8 + i].y;
                rectVs[1].z = rangeVs[iN2*8 + i].z;
                status = true;
            }
        }
        iN2 += 1;
        
        pushVec1 = {rectVs[0].x, 0, rectVs[1].z};
        oppositeSideVs.push_back(pushVec1);

        pushVec1 = {rectVs[1].x, 0, rectVs[1].z};
        oppositeSideVs.push_back(pushVec1);

        pushVec1 = {0, rectVs[0].y, rectVs[1].z};
        oppositeSideVs.push_back(pushVec1);

        pushVec1 = {0, rectVs[1].y, rectVs[1].z};
        oppositeSideVs.push_back(pushVec1);

        orizinZ.push_back(rectVs[0].z);
        oppositeZ.push_back(rectVs[1].z);
    }

    Vec3d zVec = {0, 0, -1};

    std::vector<double> rangeXyzVsCos = vec.getVecsDotCos(zVec, oppositeSideVs);

    for (int i = 0; i < rangeXyzVsCos.size() / 4; i++){
        if (orizinZ[i] >= -farZ && oppositeZ[i] <= -nearZ){
            if (rangeXyzVsCos[i*4] >= viewAngleCos.x || rangeXyzVsCos[i*4 + 1] >= viewAngleCos.x){
                if (rangeXyzVsCos[i*4 + 2] >= viewAngleCos.y || rangeXyzVsCos[i*4 + 3] >= viewAngleCos.y){
                    renderTargetObj.push_back(objOrder[i]);
                }
            }
        }
    }
}


void Camera::polyBilateralJudge(std::unordered_map<std::wstring, Object> objects){
    std::vector<Vec3d> vs;
    std::vector<Vec3d> normals;

    std::vector<int> objFaceIs;

    FaceNormals faceN;
    int iN1 = 0;
    for(int i = 0; i < renderTargetObj.size(); i++){
        for (int j = 0; j < objects[renderTargetObj[i]].poly.vId.size(); j++){
            faceN.v.push_back(
                objects[renderTargetObj[i]].v.world[
                    objects[renderTargetObj[i]].poly.vId[j].n1
                ]
            );

            faceN.normal.push_back(
                objects[renderTargetObj[i]].v.normal[
                    objects[renderTargetObj[i]].poly.normalId[j].n1
                ]
            );

            iN1 += 1;
        }

        objFaceIs.push_back(iN1);
        iN1 = 0;

        vs.insert(vs.end(), faceN.v.begin(), faceN.v.end());
        normals.insert(normals.end(), faceN.normal.begin(), faceN.normal.end());
        faceN.v.clear();
        faceN.normal.clear();
    }

    std::vector<Vec3d> cnvtVs = mt.transRotConvert(wPos, rotAngle, vs);
    std::vector<Vec3d> cnvtNs = mt.rotConvert(rotAngle, normals);

    std::vector<double> vecsCos = vec.getSameSizeVecsDotCos(cnvtVs, cnvtNs);

    int iN2 = 0;
    PolyNameInfo pushPoly;
    for (int i = 0; i < objFaceIs.size(); i++){
        for (int j = 0; j < objFaceIs[i]; j++){
            if (vecsCos[iN2 + j] <= 0){
                pushPoly.objName = renderTargetObj[i];
                pushPoly.polyId = j;

                renderTargetPoly.push_back(pushPoly);
            }
        }

        iN2 += objFaceIs[i];
    }
}


void Camera::polyCulling(
    std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource>* ptRS
){
    std::vector<Vec3d> polyVs;
    std::vector<Vec3d> polyNs;
    for (int i = 0; i < renderTargetPoly.size(); i++){
        polyVs.push_back(
            objects[renderTargetPoly[i].objName].v.world[
                objects[renderTargetPoly[i].objName].poly.vId[renderTargetPoly[i].polyId].n1
            ]
        );
        polyVs.push_back(
            objects[renderTargetPoly[i].objName].v.world[
                objects[renderTargetPoly[i].objName].poly.vId[renderTargetPoly[i].polyId].n2
            ]
        );
        polyVs.push_back(
            objects[renderTargetPoly[i].objName].v.world[
                objects[renderTargetPoly[i].objName].poly.vId[renderTargetPoly[i].polyId].n3
            ]
        );

        polyNs.push_back(
            objects[renderTargetPoly[i].objName].v.normal[
                objects[renderTargetPoly[i].objName].poly.normalId[renderTargetPoly[i].polyId].n1
            ]
        );

    }

    std::vector<Vec3d> cnvtPolyVs = mt.transRotConvert(wPos, rotAngle, polyVs);
    std::vector<Vec3d> cnvtPolyNs = mt.rotConvert(rotAngle, polyNs);

    std::vector<Vec3d> cnvt2dPolyVs;
    for (int i = 0; i < cnvtPolyVs.size(); i++){
        cnvt2dPolyVs.push_back({
            cnvtPolyVs[i].x,
            0,
            cnvtPolyVs[i].z
        });

        cnvt2dPolyVs.push_back({
            0,
            cnvtPolyVs[i].y,
            cnvtPolyVs[i].z
        });
    }

    Vec3d zVec = {0, 0, -1};

    std::vector<double> polyVCos = vec.getVecsDotCos(zVec, cnvt2dPolyVs);

    RasterizeSource pushRS;
    std::vector<PolyNameInfo> needRangeVs;
    std::vector<int> cnvtVsIndex;
    int inViewVolume = 0;
    for (int i = 0; i < renderTargetPoly.size(); i++){
        if (cnvtPolyVs[i*3].z >= -farZ && cnvtPolyVs[i*3].z <= -nearZ){
            if (polyVCos[i*6] >= viewAngleCos.x){
                if (polyVCos[i*6 + 1] >= viewAngleCos.y){
                    inViewVolume += 1;
                    pushRS.scPixelVs.wVs.push_back(cnvtPolyVs[i*3]);
                }
            }
        }

        if (cnvtPolyVs[i*3 + 1].z >= -farZ && cnvtPolyVs[i*3 + 1].z <= -nearZ){
            if (polyVCos[i*6 + 2] >= viewAngleCos.x){
                if (polyVCos[i*6 + 3] >= viewAngleCos.y){
                    inViewVolume += 1;
                    pushRS.scPixelVs.wVs.push_back(cnvtPolyVs[i*3 + 1]);
                }
            }
        }

        if (cnvtPolyVs[i*3 + 2].z >= -farZ && cnvtPolyVs[i*3 + 2].z <= -nearZ){
            if (polyVCos[i*6 + 4] >= viewAngleCos.x){
                if (polyVCos[i*6 + 5] >= viewAngleCos.y){
                    inViewVolume += 1;
                    pushRS.scPixelVs.wVs.push_back(cnvtPolyVs[i*3 + 2]);
                }
            }
        }

        if (inViewVolume != 0){
            pushRS.renderPoly.objName = renderTargetPoly[i].objName;
            pushRS.renderPoly.polyId = renderTargetPoly[i].polyId;

            pushRS.polyCamVs.push_back(cnvtPolyVs[i*3]);
            pushRS.polyCamVs.push_back(cnvtPolyVs[i*3 + 1]);
            pushRS.polyCamVs.push_back(cnvtPolyVs[i*3 + 2]);

            pushRS.polyN = cnvtPolyNs[i];

            (*ptRS).push_back(pushRS);

            if (inViewVolume != 3){
                shapeCnvtTargetI.push_back((*ptRS).size() - 1);
            }

            inViewVolume = 0;
            pushRS.polyCamVs.clear();
            pushRS.scPixelVs.wVs.clear();
        }
        else{
            needRangeVs.push_back({
                renderTargetPoly[i].objName, renderTargetPoly[i].polyId
            });

            cnvtVsIndex.push_back(i);
        }
    }

    RangeRect polyRange;
    std::vector<Vec3d> oppositeSideVs;
    std::vector<double> orizinZ;
    std::vector<double> oppositeZ;
    Vec3d pushVec;
    for (int i = 0; i < needRangeVs.size(); i++){
        polyRange.origin.x = GLPA_CAMERA_OBJ_WV_1(n1).x;
        polyRange.origin.y = GLPA_CAMERA_OBJ_WV_1(n1).y;
        polyRange.origin.z = GLPA_CAMERA_OBJ_WV_1(n1).z;

        polyRange.opposite.x = GLPA_CAMERA_OBJ_WV_1(n1).x;
        polyRange.opposite.y = GLPA_CAMERA_OBJ_WV_1(n1).y;
        polyRange.opposite.z = GLPA_CAMERA_OBJ_WV_1(n1).z;
        polyRange.status = true;

        GLPA_CAMERA_POLY_NEED_RANGE_IFS(n2);
        GLPA_CAMERA_POLY_NEED_RANGE_IFS(n3);

        pushVec = {polyRange.origin.x, 0, polyRange.opposite.z};
        oppositeSideVs.push_back(pushVec);

        pushVec = {polyRange.opposite.x, 0, polyRange.opposite.z};
        oppositeSideVs.push_back(pushVec);

        pushVec = {0, polyRange.origin.y, polyRange.opposite.z};
        oppositeSideVs.push_back(pushVec);

        pushVec = {0, polyRange.origin.y, polyRange.opposite.z};
        oppositeSideVs.push_back(pushVec);

        orizinZ.push_back(polyRange.origin.z);
        oppositeZ.push_back(polyRange.opposite.z);
    }

    std::vector<double> rangeXyzVsCos = vec.getVecsDotCos(zVec, oppositeSideVs);

    RasterizeSource pushRS2;
    for (int i = 0; i < needRangeVs.size(); i++){
        if (orizinZ[i] >= -farZ && oppositeZ[i] <= -nearZ){
            if (rangeXyzVsCos[i*4] >= viewAngleCos.x || rangeXyzVsCos[i*4 + 1] >= viewAngleCos.x){
                if (rangeXyzVsCos[i*4 + 2] >= viewAngleCos.y || rangeXyzVsCos[i*4 + 3] >= viewAngleCos.y){
                    pushRS2.renderPoly.objName = needRangeVs[i].objName;
                    pushRS2.renderPoly.polyId = needRangeVs[i].polyId;

                    pushRS2.polyCamVs.push_back(cnvtPolyVs[cnvtVsIndex[i]*3]);
                    pushRS2.polyCamVs.push_back(cnvtPolyVs[cnvtVsIndex[i]*3 + 1]);
                    pushRS2.polyCamVs.push_back(cnvtPolyVs[cnvtVsIndex[i]*3 + 2]);

                    pushRS2.polyN = cnvtPolyNs[cnvtVsIndex[i]];

                    (*ptRS).push_back(pushRS2);
                    shapeCnvtTargetI.push_back((*ptRS).size() - 1);
                    
                    pushRS2.polyCamVs.clear();
                }
            }
        }
    }
}


__global__ void glpaGpuGetPolyVvDot(
    double* polyFaceDot,
    double* vvFaceDot,
    double* polyOneVs,
    double* polyNs,
    double* vvLineStartVs,
    double* vvLineEndVs,
    double* vvOneVs,
    double* vvNs,
    double* polyLineStartVs,
    double* polyLineEndVs,
    int polyFaceAmout,
    int vvLineAmout,
    int vvFaceAmout,
    int polyLineAmout
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyFaceAmout){
        if (j < vvLineAmout){
            polyFaceDot[i*vvLineAmout*2 + j*2] = 
            (vvLineStartVs[j*3] - polyOneVs[i*3]) * polyNs[i*3] + 
            (vvLineStartVs[j*3 +1] - polyOneVs[i*3 + 1]) * polyNs[i*3 + 1] + 
            (vvLineStartVs[j*3 +2] - polyOneVs[i*3 + 2]) * polyNs[i*3 + 2];

            polyFaceDot[i*vvLineAmout*2 + j*2 + 1] = 
            (vvLineEndVs[j*3] - polyOneVs[i*3]) * polyNs[i*3] + 
            (vvLineEndVs[j*3 +1] - polyOneVs[i*3 + 1]) * polyNs[i*3 + 1] + 
            (vvLineEndVs[j*3 +2] - polyOneVs[i*3 + 2]) * polyNs[i*3 + 2];
        }
    }

    if (i >= polyFaceAmout && i < (polyFaceAmout + vvFaceAmout)){
        if (j < polyLineAmout){
            vvFaceDot[(i-polyFaceAmout)*polyLineAmout*2 + j*2] = 
            (polyLineStartVs[j*3] - vvOneVs[(i-polyFaceAmout)*3]) * vvNs[(i-polyFaceAmout)*3] + 
            (polyLineStartVs[j*3 + 1] - vvOneVs[(i-polyFaceAmout)*3 + 1]) * vvNs[(i-polyFaceAmout)*3 + 1] + 
            (polyLineStartVs[j*3 + 2] - vvOneVs[(i-polyFaceAmout)*3 + 2]) * vvNs[(i-polyFaceAmout)*3 + 2];

            vvFaceDot[(i-polyFaceAmout)*polyLineAmout*2 + j*2 + 1] = 
            (polyLineEndVs[j*3] - vvOneVs[(i-polyFaceAmout)*3]) * vvNs[(i-polyFaceAmout)*3] + 
            (polyLineEndVs[j*3 + 1] - vvOneVs[(i-polyFaceAmout)*3 + 1]) * vvNs[(i-polyFaceAmout)*3 + 1] + 
            (polyLineEndVs[j*3 + 2] - vvOneVs[(i-polyFaceAmout)*3 + 2]) * vvNs[(i-polyFaceAmout)*3 + 2];
        }
    }
}


void Camera::polyVvLineDot(std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource> *ptRS){
    polyFaceAmount = shapeCnvtTargetI.size();
    polyLineAmout = shapeCnvtTargetI.size() * 3;

    hPolyFaceDot = (double*)malloc(sizeof(double)*polyFaceAmount*vvLineAmout*2);
    hVvFaceDot = (double*)malloc(sizeof(double)*vvFaceAmout*polyLineAmout*2);

    double* hPolyOneVs = (double*)malloc(sizeof(double)*shapeCnvtTargetI.size()*3);
    double* hPolyNs = (double*)malloc(sizeof(double)*shapeCnvtTargetI.size()*3);
    double* hVvLineStartVs = (double*)malloc(sizeof(double)*viewVolume.lines.size()*3);
    double* hVvLineEndVs = (double*)malloc(sizeof(double)*viewVolume.lines.size()*3);

    for (int i = 0; i < shapeCnvtTargetI.size(); i++){
        hPolyOneVs[i*3] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].x;
        hPolyOneVs[i*3 + 1] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].y;
        hPolyOneVs[i*3 + 2] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].z;

        hPolyNs[i*3] = (*ptRS)[shapeCnvtTargetI[i]].polyN.x;
        hPolyNs[i*3 + 1] = (*ptRS)[shapeCnvtTargetI[i]].polyN.y;
        hPolyNs[i*3 + 2] = (*ptRS)[shapeCnvtTargetI[i]].polyN.z;
    }
    
    for (int i = 0; i < viewVolume.lines.size(); i++){
        hVvLineStartVs[i*3] = viewVolume.lines[i].startV.x;
        hVvLineStartVs[i*3 + 1] = viewVolume.lines[i].startV.y;
        hVvLineStartVs[i*3 + 2] = viewVolume.lines[i].startV.z;

        hVvLineEndVs[i*3] = viewVolume.lines[i].endV.x;
        hVvLineEndVs[i*3 + 1] = viewVolume.lines[i].endV.y;
        hVvLineEndVs[i*3 + 2] = viewVolume.lines[i].endV.z;
    }


    double* hVvOneVs = (double*)malloc(sizeof(double)*viewVolume.face.v.size()*3);
    double* hVvNs = (double*)malloc(sizeof(double)*viewVolume.face.normal.size()*3);
    double* hPolyLineStartVs = (double*)malloc(sizeof(double)*shapeCnvtTargetI.size()*3*3);
    double* hPolyLineEndVs = (double*)malloc(sizeof(double)*shapeCnvtTargetI.size()*3*3);

    for (int i = 0; i < viewVolume.face.v.size(); i++){
        hVvOneVs[i*3] = viewVolume.face.v[i].x;
        hVvOneVs[i*3 + 1] = viewVolume.face.v[i].y;
        hVvOneVs[i*3 + 2] = viewVolume.face.v[i].z;

        hVvNs[i*3] = viewVolume.face.normal[i].x;
        hVvNs[i*3 + 1] = viewVolume.face.normal[i].y;
        hVvNs[i*3 + 2] = viewVolume.face.normal[i].z;
    }

    for (int i = 0; i < shapeCnvtTargetI.size(); i++){
        for (int j = 0; j < 3; j++){
            hPolyLineStartVs[i*3*3 + j*3] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j].x;
            hPolyLineStartVs[i*3*3 + j*3 + 1] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j].y;
            hPolyLineStartVs[i*3*3 + j*3 + 2] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j].z;
        }

        for (int j = 0; j < 3; j++){
            if (j != 2){
                hPolyLineEndVs[i*3*3 + j*3] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j + 1].x;
                hPolyLineEndVs[i*3*3 + j*3 + 1] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j + 1].y;
                hPolyLineEndVs[i*3*3 + j*3 + 2] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[j + 1].z;
            }
            else{
                hPolyLineEndVs[i*3*3 + j*3] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].x;
                hPolyLineEndVs[i*3*3 + j*3 + 1] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].y;
                hPolyLineEndVs[i*3*3 + j*3 + 2] = (*ptRS)[shapeCnvtTargetI[i]].polyCamVs[0].z;
            }
        }
    }

    double* dPolyFaceDot;
    double* dVvFaceDot;
    cudaMalloc((void**)&dPolyFaceDot, sizeof(double)*polyFaceAmount*vvLineAmout*2);
    cudaMalloc((void**)&dVvFaceDot, sizeof(double)*vvFaceAmout*polyLineAmout*2);

    double* dPolyOneVs;
    double* dPolyNs;
    double* dVvLineStartVs;
    double* dVvLineEndVs;
    cudaMalloc((void**)&dPolyOneVs, sizeof(double)*shapeCnvtTargetI.size()*3);
    cudaMalloc((void**)&dPolyNs, sizeof(double)*shapeCnvtTargetI.size()*3);
    cudaMalloc((void**)&dVvLineStartVs, sizeof(double)*viewVolume.lines.size()*3);
    cudaMalloc((void**)&dVvLineEndVs, sizeof(double)*viewVolume.lines.size()*3);

    cudaMemcpy(dPolyOneVs, hPolyOneVs, sizeof(double)*shapeCnvtTargetI.size()*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyNs, hPolyNs, sizeof(double)*shapeCnvtTargetI.size()*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dVvLineStartVs, hVvLineStartVs, sizeof(double)*viewVolume.lines.size()*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dVvLineEndVs, hVvLineEndVs, sizeof(double)*viewVolume.lines.size()*3, cudaMemcpyHostToDevice);

    double* dVvOneVs;
    double* dVvNs;
    double* dPolyLineStartVs;
    double* dPolyLineEndVs;
    cudaMalloc((void**)&dVvOneVs, sizeof(double)*viewVolume.face.v.size()*3);
    cudaMalloc((void**)&dVvNs, sizeof(double)*viewVolume.face.normal.size()*3);
    cudaMalloc((void**)&dPolyLineStartVs, sizeof(double)*shapeCnvtTargetI.size()*3*3);
    cudaMalloc((void**)&dPolyLineEndVs, sizeof(double)*shapeCnvtTargetI.size()*3*3);

    cudaMemcpy(dVvOneVs, hVvOneVs, sizeof(double)*viewVolume.face.v.size()*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dVvNs, hVvNs, sizeof(double)*viewVolume.face.normal.size()*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyLineStartVs, hPolyLineStartVs, sizeof(double)*shapeCnvtTargetI.size()*3*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyLineEndVs, hPolyLineEndVs, sizeof(double)*shapeCnvtTargetI.size()*3*3, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid(((polyFaceAmount*vvLineAmout + vvFaceAmout*polyLineAmout) + dimBlock.x - 1) 
    / dimBlock.x, ((polyFaceAmount*vvLineAmout + vvFaceAmout*polyLineAmout) + dimBlock.y - 1) / dimBlock.y);
    glpaGpuGetPolyVvDot<<<dimGrid, dimBlock>>>(
        dPolyFaceDot,
        dVvFaceDot,
        dPolyOneVs,
        dPolyNs,
        dVvLineStartVs,
        dVvLineEndVs,
        dVvOneVs,
        dVvNs,
        dPolyLineStartVs,
        dPolyLineEndVs,
        polyFaceAmount,
        vvLineAmout,
        vvFaceAmout,
        polyLineAmout
    );
    cudaError_t error = cudaGetLastError();

    cudaMemcpy(hPolyFaceDot, dPolyFaceDot, sizeof(double)*polyFaceAmount*vvLineAmout*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVvFaceDot, dVvFaceDot, sizeof(double)*vvFaceAmout*polyLineAmout*2, cudaMemcpyDeviceToHost);

    free(hPolyOneVs);
    free(hPolyNs);
    free(hVvLineStartVs);
    free(hVvLineEndVs);

    free(hVvOneVs);
    free(hVvNs);
    free(hPolyLineStartVs);
    free(hPolyLineEndVs);

    cudaFree(dPolyFaceDot);
    cudaFree(dVvFaceDot);
    cudaFree(dVvLineStartVs);
    cudaFree(dVvLineEndVs);
    cudaFree(dPolyLineStartVs);
    cudaFree(dPolyLineEndVs);
    cudaFree(dPolyOneVs);
    cudaFree(dPolyNs);
    cudaFree(dVvOneVs);
    cudaFree(dVvNs);
}



__global__ void glpaGpuGetIntxn(
    double* polyFaceLineVs,
    double* polyFaceDot,
    double* polyFaceInxtn,

    double* vvFaceLineVs,
    double* vvFaceDot,
    double* vvFaceInxtn,

    int polyFaceSize,
    int vvFaceSize
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyFaceSize){
        if (j < 3){
            polyFaceInxtn[i*3 + j] = polyFaceLineVs[i*6 + j] + 
            (polyFaceLineVs[i*6 + j+3] - polyFaceLineVs[i*6 + j]) * (
            fabs(polyFaceDot[i*2]) / (
            fabs(polyFaceDot[i*2]) + fabs(polyFaceDot[i*2 + 1])));
        }
    }
    
    if(i >= polyFaceSize && i < (polyFaceSize + vvFaceSize)){
        if (j < 3){
            vvFaceInxtn[(i - polyFaceSize)*3 + j] = vvFaceLineVs[(i - polyFaceSize)*6 + j] + 
            (vvFaceLineVs[(i - polyFaceSize)*6 + j+3] - vvFaceLineVs[(i - polyFaceSize)*6 + j]) * (
            fabs(vvFaceDot[(i - polyFaceSize)*2]) / (
            fabs(vvFaceDot[(i - polyFaceSize)*2]) + fabs(vvFaceDot[(i - polyFaceSize)*2 + 1])));
        }
    }
}


__global__ void glpaGpuGetIACos(
    double* polyFaceVs,
    double* polyFaceInxtn,
    double* polyFaceIaCos,
    double* vvFaceVs,
    double* vvFaceInxtn,
    double* vvFaceIaCos,
    int polyFaceSize,
    int vvFaceSize
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < polyFaceSize){
        if (j < 3){
            polyFaceIaCos[i*3*2 + j*2] = 
            ((polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceInxtn[i*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceInxtn[i*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceInxtn[i*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2])) 
            /
            (sqrt((polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]))
            * 
            sqrt((polyFaceInxtn[i*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceInxtn[i*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceInxtn[i*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceInxtn[i*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceInxtn[i*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceInxtn[i*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2])));

            polyFaceIaCos[i*3*2 + j*2 + 1] = 
            ((polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2])) 
            /
            (sqrt((polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceVs[i*5*3 + ((j+1)+1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]))
            * 
            sqrt((polyFaceVs[i*5*3 + ((j+1)-1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3] - polyFaceVs[i*5*3 + (j+1)*3]) + 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 1] - polyFaceVs[i*5*3 + (j+1)*3 + 1]) + 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2]) * 
            (polyFaceVs[i*5*3 + ((j+1)-1)*3 + 2] - polyFaceVs[i*5*3 + (j+1)*3 + 2])));

        }
    }
    
    if(i >= polyFaceSize && i < (polyFaceSize + vvFaceSize)){
        if (j < 4){
            vvFaceIaCos[(i-polyFaceSize)*4*2 + j*2] = 
            ((vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2])) 
            /
            (sqrt((vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]))
            * 
            sqrt((vvFaceInxtn[(i-polyFaceSize)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceInxtn[(i-polyFaceSize)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2])));

            vvFaceIaCos[(i-polyFaceSize)*4*2 + j*2 + 1] = 
            ((vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2])) 
            /
            (sqrt((vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)+1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]))
            * 
            sqrt((vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 1] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 1]) + 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2]) * 
            (vvFaceVs[(i-polyFaceSize)*6*3 + ((j+1)-1)*3 + 2] - vvFaceVs[(i-polyFaceSize)*6*3 + (j+1)*3 + 2])));
        }
    }
}


void Camera::inxtnInteriorAngle(std::vector<RasterizeSource>* ptRS){
    polyFaceAmount = shapeCnvtTargetI.size();
    polyLineAmout = shapeCnvtTargetI.size() * 3;

    std::vector<double> polyFaceVs; // 0.x 0.y 0.z 1.x 1.y 1.z 2.x 2.y 2.z n*9
    std::vector<double> polyFaceLineVs; // start.x start.y start.z end.x end.y end.z n*6
    std::vector<double> polyFaceDot; // startVdot endVdot n*2

    std::vector<double> vvFaceVs; // 0.x 0.y 0.z 1.x 1.y 1.z 2.x 2.y 2.z 3.x 3.y 3.z n*9
    std::vector<double> vvFaceLineVs; // start.x start.y start.z end.x end.y end.z n*6
    std::vector<double> vvFaceDot; // startVdot endVdot n*2

    bool faceIExist = false;
    for (int i = 0; i < polyFaceAmount; i++){

        for (int j = 0; j < vvLineAmout; j++){
            if (hPolyFaceDot[i*vvLineAmout*2 + j*2] == 0){
                (*ptRS)[shapeCnvtTargetI[i]].scPixelVs.wVs.push_back(viewVolume.lines[j].startV);
                faceIExist = true;
            }
            if (hPolyFaceDot[i*vvLineAmout*2 + j*2 + 1] == 0){
                (*ptRS)[shapeCnvtTargetI[i]].scPixelVs.wVs.push_back(viewVolume.lines[j].endV);
                faceIExist = true;
            }

            if (faceIExist){
                faceIExist = false;
                continue;
            }

            if (
                (hPolyFaceDot[i*vvLineAmout*2 + j*2] > 0 && hPolyFaceDot[i*vvLineAmout*2 + j*2 + 1] < 0) ||
                (hPolyFaceDot[i*vvLineAmout*2 + j*2] < 0 && hPolyFaceDot[i*vvLineAmout*2 + j*2 + 1] > 0)
            ){
                polyRsI.push_back(shapeCnvtTargetI[i]);
                vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[i]].polyCamVs, &polyFaceVs, 2);
                vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[i]].polyCamVs, &polyFaceVs, 0);
                vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[i]].polyCamVs, &polyFaceVs, 1);
                vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[i]].polyCamVs, &polyFaceVs, 2);
                vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[i]].polyCamVs, &polyFaceVs, 0);

                polyFaceLineVs.push_back(viewVolume.lines[j].startV.x);
                polyFaceLineVs.push_back(viewVolume.lines[j].startV.y);
                polyFaceLineVs.push_back(viewVolume.lines[j].startV.z);
                polyFaceLineVs.push_back(viewVolume.lines[j].endV.x);
                polyFaceLineVs.push_back(viewVolume.lines[j].endV.y);
                polyFaceLineVs.push_back(viewVolume.lines[j].endV.z);

                polyFaceDot.push_back(hPolyFaceDot[i*vvLineAmout*2 + j*2]);
                polyFaceDot.push_back(hPolyFaceDot[i*vvLineAmout*2 + j*2 + 1]);

                continue;
            }
        }
    }


    for (int i = 0; i < vvFaceAmout; i++){
        for (int j = 0; j < polyFaceAmount; j++){
            for (int k = 0; k < 3; k++){
                if (hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2] == 0){
                    (*ptRS)[shapeCnvtTargetI[j]].scPixelVs.wVs.push_back((*ptRS)[shapeCnvtTargetI[j]].polyCamVs[k]);
                    faceIExist = true;
                }

                if (hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2 + 1] == 0){
                    if (j != 2){
                        (*ptRS)[shapeCnvtTargetI[j]].scPixelVs.wVs.push_back(
                            (*ptRS)[shapeCnvtTargetI[j]].polyCamVs[k + 1]
                        );
                    } 
                    else{
                        (*ptRS)[shapeCnvtTargetI[j]].scPixelVs.wVs.push_back((*ptRS)[shapeCnvtTargetI[j]].polyCamVs[0]);
                    }
                    faceIExist = true;
                }

                if (faceIExist){
                    faceIExist = false;
                    continue;
                }

                if (
                    (hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2] > 0 && 
                    hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2 + 1] < 0) ||
                    (hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2] < 0 && 
                    hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2 + 1] > 0)
                ){
                    vvRsI.push_back(shapeCnvtTargetI[j]);
                    viewVolume.pushFaceVsToDouble(&vvFaceVs, i);

                    vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[j]].polyCamVs, &vvFaceLineVs, k);
                    if (k != 2){
                        vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[j]].polyCamVs, &vvFaceLineVs, k + 1);
                    } 
                    else{
                        vec.pushVecToDouble((*ptRS)[shapeCnvtTargetI[j]].polyCamVs, &vvFaceLineVs, 0);
                    }

                    vvFaceDot.push_back(hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2]);
                    vvFaceDot.push_back(hVvFaceDot[i*polyFaceAmount*6 + 6*j + k*2 + 1]);

                    continue;
                }
            }
        }
    }

    double* hPolyFaceLineVs = (double*)malloc(sizeof(double)*polyFaceLineVs.size());
    double* hCalcPolyFaceDot = (double*)malloc(sizeof(double)*polyFaceDot.size());
    hPolyFaceInxtn = (double*)malloc(sizeof(double)*polyRsI.size()*3);

    double* hVvFaceLineVs = (double*)malloc(sizeof(double)*vvFaceLineVs.size());
    double* hCalcVvFaceDot = (double*)malloc(sizeof(double)*vvFaceDot.size());
    hVvFaceInxtn = (double*)malloc(sizeof(double)*vvRsI.size()*3);

    memcpy(hPolyFaceLineVs, polyFaceLineVs.data(), sizeof(double)*polyFaceLineVs.size());
    memcpy(hCalcPolyFaceDot, polyFaceDot.data(), sizeof(double)*polyFaceDot.size());

    memcpy(hVvFaceLineVs, vvFaceLineVs.data(), sizeof(double)*vvFaceLineVs.size());
    memcpy(hCalcVvFaceDot, vvFaceDot.data(), sizeof(double)*vvFaceDot.size());

    double* dPolyFaceLineVs;
    double* dPolyFaceDot;
    double* dPolyFaceInxtn;

    double* dVvFaceLineVs;
    double* dVvFaceDot;
    double* dVvFaceInxtn;

    cudaMalloc((void**)&dPolyFaceLineVs, sizeof(double)*polyFaceLineVs.size());
    cudaMalloc((void**)&dPolyFaceDot, sizeof(double)*polyFaceDot.size());
    cudaMalloc((void**)&dPolyFaceInxtn, sizeof(double)*polyRsI.size()*3);

    cudaMalloc((void**)&dVvFaceLineVs, sizeof(double)*vvFaceLineVs.size());
    cudaMalloc((void**)&dVvFaceDot, sizeof(double)*vvFaceDot.size());
    cudaMalloc((void**)&dVvFaceInxtn, sizeof(double)*vvRsI.size()*3);

    cudaMemcpy(dPolyFaceLineVs, hPolyFaceLineVs, sizeof(double)*polyFaceLineVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dPolyFaceDot, hCalcPolyFaceDot, sizeof(double)*polyFaceDot.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(dVvFaceLineVs, hVvFaceLineVs, sizeof(double)*vvFaceLineVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dVvFaceDot, hCalcVvFaceDot, sizeof(double)*vvFaceDot.size(), cudaMemcpyHostToDevice);


    int inxtnAmount = polyRsI.size() * 3 + vvRsI.size() * 3;
    dim3 dimBlock(32, 32);
    dim3 dimGrid((inxtnAmount + dimBlock.x - 1) 
    / dimBlock.x, (inxtnAmount + dimBlock.y - 1) / dimBlock.y);
    glpaGpuGetIntxn<<<dimGrid, dimBlock>>>(
        dPolyFaceLineVs,
        dPolyFaceDot,
        dPolyFaceInxtn,
        dVvFaceLineVs,
        dVvFaceDot,
        dVvFaceInxtn,
        polyRsI.size(),
        vvRsI.size()
    );
    cudaError_t error = cudaGetLastError();
    cudaMemcpy(hPolyFaceInxtn, dPolyFaceInxtn, sizeof(double)*polyRsI.size()*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVvFaceInxtn, dVvFaceInxtn, sizeof(double)*vvRsI.size()*3, cudaMemcpyDeviceToHost);

    free(hPolyFaceLineVs);
    free(hPolyFaceDot);
    free(hCalcPolyFaceDot);
    free(hVvFaceLineVs);
    free(hVvFaceDot);
    free(hCalcVvFaceDot);

    cudaFree(dPolyFaceLineVs);
    cudaFree(dPolyFaceDot);
    cudaFree(dVvFaceLineVs);
    cudaFree(dVvFaceDot);

    double* hPolyFaceVs = (double*)malloc(sizeof(double)*polyFaceVs.size());
    hPolyFaceIACos = (double*)malloc(sizeof(double)*polyRsI.size()*6);
    memcpy(hPolyFaceVs, polyFaceVs.data(), sizeof(double)*polyFaceVs.size());

    double* hVvFaceVs = (double*)malloc(sizeof(double)*vvFaceVs.size());
    hVvFaceIACos = (double*)malloc(sizeof(double)*vvRsI.size()*8);
    memcpy(hVvFaceVs, vvFaceVs.data(), sizeof(double)*vvFaceVs.size());

    double* dPolyFaceVs;
    double* dPolyFaceIACos;
    cudaMalloc((void**)&dPolyFaceVs, sizeof(double)*polyFaceVs.size());
    cudaMalloc((void**)&dPolyFaceIACos, sizeof(double)*polyRsI.size()*6);
    cudaMemcpy(dPolyFaceVs, hPolyFaceVs, sizeof(double)*polyFaceVs.size(), cudaMemcpyHostToDevice);

    double* dVvFaceVs;
    double* dVvFaceIACos;
    cudaMalloc((void**)&dVvFaceVs, sizeof(double)*vvFaceVs.size());
    cudaMalloc((void**)&dVvFaceIACos, sizeof(double)*vvRsI.size()*8);
    cudaMemcpy(dVvFaceVs, hVvFaceVs, sizeof(double)*vvFaceVs.size(), cudaMemcpyHostToDevice);

    glpaGpuGetIACos<<<dimGrid, dimBlock>>>(
        dPolyFaceVs,
        dPolyFaceInxtn,
        dPolyFaceIACos,
        dVvFaceVs,
        dVvFaceInxtn,
        dVvFaceIACos,
        polyRsI.size(),
        vvRsI.size()
    );
    cudaError_t error2 = cudaGetLastError();
    cudaMemcpy(hPolyFaceIACos, dPolyFaceIACos, sizeof(double)*polyRsI.size()*6, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVvFaceIACos, dVvFaceIACos, sizeof(double)*vvRsI.size()*8, cudaMemcpyDeviceToHost);

    free(hPolyFaceVs);
    free(hVvFaceVs);

    cudaFree(dPolyFaceVs);
    cudaFree(dPolyFaceInxtn);
    cudaFree(dPolyFaceIACos);

    cudaFree(dVvFaceVs);
    cudaFree(dVvFaceInxtn);
    cudaFree(dVvFaceIACos);

}


void Camera::setPolyInxtn(
    std::unordered_map<std::wstring, Object> objects, std::vector<RasterizeSource> *ptRS
){
    for (int i = 0; i < polyRsI.size(); i++){
        if (
            (hPolyFaceIACos[i*6] >= hPolyFaceIACos[i*6 + 1]) &&
            (hPolyFaceIACos[i*6 + 2] >= hPolyFaceIACos[i*6 + 3]) &&
            (hPolyFaceIACos[i*6 + 4] >= hPolyFaceIACos[i*6 + 5]) 
        ){
            (*ptRS)[polyRsI[i]].scPixelVs.wVs.push_back({
                hPolyFaceInxtn[i*3],
                hPolyFaceInxtn[i*3 + 1],
                hPolyFaceInxtn[i*3 + 2]
            });
        }
    }

    for (int i = 0; i < vvRsI.size(); i++){
        if (
            (hVvFaceIACos[i*8] >= hVvFaceIACos[i*8 + 1]) &&
            (hVvFaceIACos[i*8 + 2] >= hVvFaceIACos[i*8 + 3]) &&
            (hVvFaceIACos[i*8 + 4] >= hVvFaceIACos[i*8 + 5]) &&
            (hVvFaceIACos[i*8 + 6] >= hVvFaceIACos[i*8 + 7])
        ){
            (*ptRS)[vvRsI[i]].scPixelVs.wVs.push_back({
                hVvFaceInxtn[i*3],
                hVvFaceInxtn[i*3 + 1],
                hVvFaceInxtn[i*3 + 2]
            });

        }
    }


    free(hPolyFaceInxtn);
    free(hVvFaceInxtn);

    free(hPolyFaceIACos);
    free(hVvFaceIACos);


}


__global__ void glpaGpuScPixelConvert(
    double* wVs, 
    double* nearZ, 
    double* farZ, 
    double* nearScSize, 
    double* scPixelSize,
    double* resultVs, 
    int wVsAmount){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < wVsAmount){
        if (j == 0){
            resultVs[i*2 + j] = 
            std::round((((wVs[i*3 + j] * -nearZ[0] / wVs[i*3 + 2]) + nearScSize[j]) /
            (nearScSize[j] * 2)) * scPixelSize[j]);
        }
        else if (j == 1){
            resultVs[i*2 + j] = 
            std::round(scPixelSize[j] - (((wVs[i*3 + j] * -nearZ[0] / wVs[i*3 + 2]) + nearScSize[j]) /
            (nearScSize[j] * 2)) * scPixelSize[j]);
        }
    }
}


void Camera::scPixelConvert(std::vector<RasterizeSource> *ptRS){
    std::vector<double> wVs;
    std::vector<int> wVsSize;
    int wVsAmount = 0;

    for (int i = 0; i < (*ptRS).size(); i++){
        wVsSize.push_back((*ptRS)[i].scPixelVs.wVs.size());

        for (int j = 0; j < (*ptRS)[i].scPixelVs.wVs.size(); j++){
            vec.pushVecToDouble((*ptRS)[i].scPixelVs.wVs, &wVs, j);
            wVsAmount += 1;
        }
    }

    double* hWvs = (double*)malloc(sizeof(double)*wVsAmount*3);
    double* hNearZ = &nearZ;
    double* hFarZ = &farZ;
    double* hNearScSize = (double*)malloc(sizeof(double)*2);
    double* hScPixelSize = (double*)malloc(sizeof(double)*2);
    double* hResultVs = (double*)malloc(sizeof(double)*wVsAmount*2);

    memcpy(hWvs, wVs.data(), sizeof(double)*wVsAmount*3);
    hNearScSize[0] = nearScrSize.x;
    hNearScSize[1] = nearScrSize.y;

    hScPixelSize[0] = scPixelSize.x;
    hScPixelSize[1] = scPixelSize.y;

    double* dWvs;
    double* dNearZ;
    double* dFarZ;
    double* dNearScSize;
    double* dScPixelSize;
    double* dResultVs;
    cudaMalloc((void**)&dWvs, sizeof(double)*wVsAmount*3);
    cudaMalloc((void**)&dNearZ, sizeof(double)*1);
    cudaMalloc((void**)&dFarZ, sizeof(double)*1);
    cudaMalloc((void**)&dNearScSize, sizeof(double)*2);
    cudaMalloc((void**)&dScPixelSize, sizeof(double)*2);
    cudaMalloc((void**)&dResultVs, sizeof(double)*wVsAmount*2);

    cudaMemcpy(dWvs, hWvs, sizeof(double)*wVsAmount*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dNearZ, hNearZ, sizeof(double)*1, cudaMemcpyHostToDevice);
    cudaMemcpy(dFarZ, hFarZ, sizeof(double)*1, cudaMemcpyHostToDevice);
    cudaMemcpy(dNearScSize, hNearScSize, sizeof(double)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dScPixelSize, hScPixelSize, sizeof(double)*2, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 32);
    dim3 dimGrid((wVsAmount*2 + dimBlock.x - 1) 
    / dimBlock.x, (wVsAmount*2 + dimBlock.y - 1) / dimBlock.y);
    glpaGpuScPixelConvert<<<dimGrid, dimBlock>>>(
        dWvs, dNearZ, dFarZ, dNearScSize, dScPixelSize, dResultVs, wVsAmount
    );
    cudaError_t error = cudaGetLastError();

    cudaMemcpy(hResultVs, dResultVs, sizeof(double)*wVsAmount*2, cudaMemcpyDeviceToHost);

    int wVsI = 0;
    for (int i = 0; i < (*ptRS).size(); i++){
        if (wVsSize[i] > 3){
            sortTargetI.push_back(i);
        }
        
        for (int j = 0; j < wVsSize[i]; j++){
            if (wVsSize[i] <= 2){
                throw std::runtime_error(ERROR_CAMERA_CANT_RASTERIZE);
            }
            else if(wVsSize[i] == 3){
                (*ptRS)[i].scPixelVs.sortedWVs.push_back({
                    hResultVs[wVsI*2],
                    hResultVs[wVsI*2 + 1]
                });
            }
            else if (wVsSize[i] > 3){
                (*ptRS)[i].scPixelVs.vs.push_back({
                    hResultVs[wVsI*2],
                    hResultVs[wVsI*2 + 1]
                });
            }

            wVsI += 1;
        }
    }

    free(hWvs);
    free(hNearScSize);
    free(hScPixelSize);
    free(hResultVs);

    cudaFree(dWvs);
    cudaFree(dNearZ);
    cudaFree(dFarZ);
    cudaFree(dNearScSize);
    cudaFree(dScPixelSize);
    cudaFree(dResultVs);

}
