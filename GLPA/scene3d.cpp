#include "scene3d.h"


void Scene3d::storeUseWndParam(int width, int height, int dpi){
    useWndWidth = width;
    useWndHeight = height;
    useWndDpi = dpi;
}


void Scene3d::edit(HDC hBufDC, LPDWORD lpPixel){
    for (auto it : sceneFrameFunc){
        it.second(hBufDC, lpPixel);
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
