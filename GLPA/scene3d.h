#ifndef SCENE3D_H_
#define SCENE3D_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>

#include "scene2d.h"
#include "error.h"


class Scene3d
{
public :
    void storeUseWndParam(int width, int height, int dpi);

    void loadObj();
    void loadMtl();
    void loadField();
    void loadSky();
    void loadLight();
    void loadCam();

    void edit(HDC hBufDC, LPDWORD lpPixel);
    void update();
    void reload();

    void releaseObj();
    void releaseMtl();
    void releaseField();
    void releaseSky();
    void releaseLight();
    void releaseCam();
    void release();

    void addSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL add_func);
    void editSceneFrameFunc(std::wstring func_name, GLPA_SCENE_FUNC_FUNCTIONAL edited_func);
    void releaseSceneFrameFunc(std::wstring func_name);

    int useWndWidth = 0;
    int useWndHeight = 0;
    int useWndDpi = 0;

private :
    std::unordered_map<std::wstring, GLPA_SCENE_FUNC_FUNCTIONAL> sceneFrameFunc;


};

#endif  SCENE3D_H_