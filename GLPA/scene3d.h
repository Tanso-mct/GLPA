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
#include "cg.h"

#include "object.h"
#include "camera.cuh"

#include "render.cuh"
#include "buffer_3d.cuh"


class Scene3d{
public :
    void initialize();
    void storeUseWndParam(int width, int height, int dpi);

    void loadCam(std::wstring cam_name);
    void loadObj(std::string scene_folder_path, std::wstring object_folder_name, std::wstring file_name);
    void loadMtl();
    void loadField();
    void loadSky();
    void loadLight();

    void selectUseCam(std::wstring cam_name);

    void setUseCamTrans(Vec3d pos_vec, Vec3d rot_vec);
    void moveUseCam(Vec3d diff_move_vec);
    void rotUseCam(Vec3d diff_rot_vec);
    

    void edit(HDC h_buffer_dc, LPDWORD lp_pixel);
    void update(HDC h_buffer_dc, LPDWORD lp_pixel);
    void reload();

    void releaseCam();
    void releaseObj(std::wstring object_folder_name, std::string file_name);
    void releaseMtl();
    void releaseField();
    void releaseSky();
    void releaseLight();
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
    std::unordered_map<std::wstring, Camera> cams;

    std::vector<RasterizeSource> rasterizeSource;

    double* zBuffRSIs = nullptr;
    double* zBuffCamVs = nullptr;
    double* zBuffComp = nullptr;

    Buffer3d buf3d;
    Render render;


};

#endif  SCENE3D_H_