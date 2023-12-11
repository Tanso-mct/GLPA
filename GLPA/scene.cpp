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
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::string narrowFolderPath = converter.to_bytes(folderPath);
    std::string narrowName;
    std::size_t lastPeriod;
    std::string extension;

    std::size_t lastSolid = narrowFolderPath.rfind("/");
    std::string narrowCuttedFolderPath = narrowFolderPath.substr(0, lastSolid);

    std::string sceneFolderName = GLPA_SCENE_FOLDER_NAME;
    std::size_t sceneFolderNamePos = narrowFolderPath.rfind(sceneFolderName);
    std::string groupName = narrowFolderPath.substr(
        sceneFolderNamePos+sceneFolderName.size(), lastSolid - (sceneFolderNamePos+sceneFolderName.size())
    );

    if (names[scName] == GLPA_SCENE_2D){
        for (auto name : fileNames){
            narrowName = converter.to_bytes(name);

            lastPeriod = narrowName.rfind(".");
            extension = narrowName.substr(lastPeriod+1, narrowName.size()-1);

            if (extension == "png"){
                data2d[scName].loadPng(narrowCuttedFolderPath, groupName, narrowName);
            }
            
        }
    }
    else if (names[scName] == GLPA_SCENE_3D){
        for (auto name : fileNames){
            narrowName = converter.to_bytes(name);

            lastPeriod = narrowName.rfind(".");
            extension = narrowName.substr(lastPeriod+1, narrowName.size()-1);

            if (extension == "obj"){

            }
            
        }
    }
}
