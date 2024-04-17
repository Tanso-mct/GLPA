#include "scene3d.h"


void Scene3d::initialize(){
    rasterizeSource.clear();
    delete[] zBuffRSIs;
    delete[] zBuffCamVs;
    delete[] zBuffComp;
}

void Scene3d::storeUseWndParam(int width, int height, int dpi){
    useWndWidth = width;
    useWndHeight = height;
    useWndDpi = dpi;
}


void Scene3d::loadCam(std::wstring camName){
    if (cams.find(camName) == cams.end()){
        useCamName = camName;


        //TODO: カメラデータをファイルから読み込めるようにする。（FBXファイルを使用する必要がある可能性がある）

        cams[camName].load(
            camName,
            {0, 0, 0},
            {0, 0, 0},
            1,
            1000,
            80,
            {16, 9},
            {(double)useWndWidth, (double)useWndHeight}
        );
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_3D_EXIST_CAM);
    }
}


void Scene3d::loadObj(std::string scFolderPass, std::wstring objFolderName, std::wstring fileName)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::string folderPass = scFolderPass + "/" + converter.to_bytes(objFolderName);
    std::wstring assignName = fileName.substr(0, fileName.find(L"."));
    objects[assignName].name = assignName;
    objects[assignName].load(fileName, folderPass);
}

void Scene3d::selectUseCam(std::wstring camName){
    if (cams.find(camName) != cams.end()){
        useCamName = camName;
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_3D_NOT_EXIST_CAM);
    }
}

void Scene3d::setUseCamTrans(Vec3d pos, Vec3d rot){
    cams[useCamName].setTrans(pos, rot);
}

void Scene3d::moveUseCam(Vec3d diffMoveVec){
    cams[useCamName].move(diffMoveVec);
}

void Scene3d::rotUseCam(Vec3d diffRotVec){
    cams[useCamName].rot(diffRotVec);
}

void Scene3d::edit(HDC hBufDC, LPDWORD lpPixel){
    for (auto it : sceneFrameFunc){
        it.second(hBufDC, lpPixel);
    }
}

void Scene3d::update(HDC hBufDC, LPDWORD lpPixel){
    cams[useCamName].defineViewVolume();

    render.prepareObjs(objects, cams[useCamName]);

    // cams[useCamName].objCulling(objects);
    // cams[useCamName].polyBilateralJudge(objects);
    // cams[useCamName].polyCulling(objects, &rasterizeSource);
    // cams[useCamName].polyVvLineDot(objects, &rasterizeSource);
    // cams[useCamName].inxtnInteriorAngle(&rasterizeSource);
    // cams[useCamName].setPolyInxtn(objects, &rasterizeSource);
    // cams[useCamName].scPixelConvert(&rasterizeSource);
    // cams[useCamName].sortScPixelVs(&rasterizeSource);
    // cams[useCamName].zBuffer(&rasterizeSource, zBuffRSIs, zBuffCamVs, zBuffComp);

    // buf3d.initialize({(double)useWndWidth, (double)useWndHeight}, useWndDpi);
    // buf3d.drawZBuff(lpPixel, zBuffComp);

    // initialize();
    // cams[useCamName].initialize();

}


void Scene3d::releaseObj(std::wstring objFolderName, std::string fileName){
    if (objects.find(objFolderName) != objects.end()){
        objects.erase(objFolderName);
    }
    else{
        std::runtime_error(ERROR_MESH_LOAD_RELEASE);
    }
}


void Scene3d::release(){
    
}


void Scene3d::addSceneFrameFunc(std::wstring funcName, GLPA_SCENE_FUNC_FUNCTIONAL addFunc){
    sceneFrameFunc[funcName] = addFunc;
}


void Scene3d::editSceneFrameFunc(std::wstring funcName, GLPA_SCENE_FUNC_FUNCTIONAL editedFunc){
    if (sceneFrameFunc.find(funcName) != sceneFrameFunc.end()){
        sceneFrameFunc[funcName] = editedFunc;
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_3D_FRAME_FUNC_NAME);
    }
    
}


void Scene3d::releaseSceneFrameFunc(std::wstring funcName){
    if (sceneFrameFunc.find(funcName) != sceneFrameFunc.end()){
        sceneFrameFunc.erase(funcName);
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_3D_FRAME_FUNC_NAME);
    }
}
