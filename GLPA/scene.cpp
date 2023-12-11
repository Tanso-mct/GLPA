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
    std::string narrowFolderPath;
    std::string narrowName;
    std::size_t lastPeriod;
    std::string extension;
    std::size_t lastSolid;
    std::string groupName;
    if (names[scName] == GLPA_SCENE_2D){
        for (auto name : fileNames){
            narrowFolderPath = converter.to_bytes(folderPath);
            narrowName = converter.to_bytes(name);

            lastPeriod = narrowName.rfind(".");
            extension = narrowName.substr(lastPeriod+1, narrowName.size()-1);

            lastSolid = narrowFolderPath.rfind("/");
            groupName = narrowFolderPath.substr(lastSolid+1, narrowName.size()-1);

            if (extension == "png"){
                data2d[scName].loadPng(narrowFolderPath, groupName, narrowName);
            }
            
        }
    }
    else if (names[scName] == GLPA_SCENE_3D){
        for (auto name : fileNames){
            std::size_t lastPeriod = name.rfind(L".");

            std::wstring extension = name.substr(lastPeriod+1, name.size()-1);

            if (extension == L"obj"){

            }
            
        }
    }
}
