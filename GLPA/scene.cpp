#include "scene.h"

void Scene::create( std::string scName, int selectType){
    if (selectType == GLPA_SCENE_2D){
        Scene2d tempScene2d;
        names.emplace(scName, GLPA_SCENE_2D);
        data2d.emplace(scName, tempScene2d);
    }
    else if (selectType == GLPA_SCENE_3D){
        Scene3d tempScene3d;
        names.emplace(scName, GLPA_SCENE_3D);
        data3d.emplace(scName, tempScene3d);
    }
    else {
        throw std::runtime_error(ERROR_SCENE_CREATE);
    }
}

void Scene::load(std::string scName, LPCWSTR folderPath, std::vector<std::wstring> fileNames){
    if (names[scName] == GLPA_SCENE_2D){
        for (auto file : fileNames){
            std::size_t lastPeriod = file.rfind(L".");

            std::wstring extension = file.substr(lastPeriod+1, file.size()-1);

            if (extension == L"png"){

            }
            
        }
    }
    else if (names[scName] == GLPA_SCENE_3D){
        for (auto file : fileNames){
            std::size_t lastPeriod = file.rfind(L".");

            std::wstring extension = file.substr(lastPeriod+1, file.size()-1);

            if (extension == L"obj"){

            }
            
        }
    }
}
