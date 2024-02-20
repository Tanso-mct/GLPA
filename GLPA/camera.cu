#include "camera.cuh"


void Camera::load(
    std::wstring argName, 
    Vec3d argWPos, 
    Vec3d argRotAngle, 
    double argNearZ, 
    double argFarZ, 
    double argViewAngle, 
    Vec2d argAspectRatio
){
    name = argName;
    wPos = argWPos;
    rotAngle = argRotAngle;
    nearZ = -argNearZ;
    farZ = -argFarZ;
    viewAngle = argViewAngle;
    aspectRatio = argAspectRatio;

    reload = true;
}


void Camera::defineViewVolume(){
    // If no reloading has been done and no changes have been made to the definition, no processing is done.
    if (!reload){
        return;
    }

    // Get screen size
    nearScrSize.x = tan(viewAngle / 2 * PI / 180) * nearZ * 2;
    nearScrSize.y = nearScrSize.x * aspectRatio.y / aspectRatio.x;

    farScrSize.x = nearScrSize.x / 2 * farZ / nearZ;
    farScrSize.y = farScrSize.x * aspectRatio.y / aspectRatio.x;

    viewAngleCos.x = cos(RAD(viewAngle));
    viewAngleCos.y = cos(nearZ / sqrt(nearZ*nearZ + (nearScrSize.y/2) * (nearScrSize.y/2)));

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

    std::vector<double> rangeXzVsCos = vec.getVecsDotCos(zVec, oppositeSideVs);

    for (int i = 0; i < rangeXzVsCos.size() / 4; i++){
        if (orizinZ[i] >= farZ && oppositeZ[i] <= nearZ){
            if (rangeXzVsCos[i*4] >= viewAngleCos.x || rangeXzVsCos[i*4 + 1] >= viewAngleCos.x){
                if (rangeXzVsCos[i*4 + 2] >= viewAngleCos.y || rangeXzVsCos[i*4 + 3] >= viewAngleCos.y){
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

    std::vector<Vec3d> convertedVs = mt.transRotConvert(wPos, rotAngle, vs);
    std::vector<Vec3d> convertedNs = mt.rotConvert(rotAngle, normals);

    std::vector<double> vecsCos = vec.getSameSizeVecsDotCos(convertedVs, convertedNs);

    int iN2 = 0;
    PolyNameInfo pushPoly;
    for (int i = 0; i < objFaceIs.size(); i++){
        for (int j = 0; j < objFaceIs[i]; j++){
            if (vecsCos[iN2 + j] >= 0){
                pushPoly.objName = renderTargetObj[i];
                pushPoly.polyId = j;

                renderTargetPoly.push_back(pushPoly);
            }
        }

        iN2 += objFaceIs[i];
    }
}

