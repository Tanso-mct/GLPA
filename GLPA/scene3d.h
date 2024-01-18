/**
 * @file scene3d.h
 * @brief
 * 日本語 : 3次元を描画するためのシーンデータクラス。
 * English : Scene data class for rendering 3D.
 * @author Tanso
 * @date 2023-11
*/


#ifndef SCENE3D_H_
#define SCENE3D_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <functional>
#include <locale>
#include <codecvt>

#include "scene2d.h"
#include "error.h"

#include "object.h"
// #include "camera.h"


class Scene3d{
public :
    void storeUseWndParam(int width, int height, int dpi);

    void loadObj(std::string scene_folder_path, std::wstring object_folder_name, std::string file_name);
    void loadMtl();
    void loadField();
    void loadSky();
    void loadLight();
    void loadCam();

    void selectUseCam();
    void editCam();

    void edit(HDC hBufDC, LPDWORD lpPixel);
    void update();
    void reload();

    void releaseObj(std::wstring object_folder_name, std::string file_name);
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
    std::wstring useCamName;

    std::unordered_map<std::wstring, Object> objects;
    // std::unordered_map<std::wstring, Camera> cameras;


};

#endif  SCENE3D_H_