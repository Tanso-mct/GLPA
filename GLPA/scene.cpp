#include "scene.h"

void Scene::setFolderPass(std::wstring scNameFolderPass){
    std::size_t lastSolid =  scNameFolderPass.rfind(L"/");
    folderPass = scNameFolderPass.substr(0, lastSolid);
}

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

void Scene::load(
    std::string scName, 
    std::wstring folderPath, 
    std::unordered_map<std::wstring, std::vector<std::wstring>> allData
){
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::string narrowFolderPath = converter.to_bytes(folderPath);
    std::string narrowName;
    std::size_t lastPeriod;
    std::string extension;

    std::unordered_map<int, std::string> tempMap;

    if (names[scName] == GLPA_SCENE_2D){
        for (auto group : allData){
            if (group.first.find(GLPA_SCENE_GROUP_NAME_L) == std::wstring::npos){
                throw std::runtime_error(ERROR_SCENE2D_LOADPNG);
            }

            data2d[scName].groupOrder.emplace(
                converter.to_bytes(group.first.substr(
                    0, 
                    group.first.find(GLPA_SCENE_GROUP_NAME_L)
                )),
                std::stod(
                    converter.to_bytes(group.first.substr(group.first.find(GLPA_SCENE_GROUP_NAME_L) + GLPA_SCENE_GROUP_NAME_L_SIZE, 
                    group.first.size()))
                )
            );
            
            for (auto pngName : group.second){
                narrowName = converter.to_bytes(pngName);

                lastPeriod = narrowName.rfind(".");
                extension = narrowName.substr(lastPeriod+1, narrowName.size()-1);

                if (extension == "png"){
                    data2d[scName].loadPng(
                        narrowFolderPath + "/" + converter.to_bytes(group.first), 
                        converter.to_bytes(group.first.substr(
                            0, 
                            group.first.find(GLPA_SCENE_GROUP_NAME_L)
                        )),
                        narrowName
                    );
                }
                //TODO: Allow text to be read from a file.
            }
        }

        data2d[scName].loadText();
    }
    else if (names[scName] == GLPA_SCENE_3D){
        for (auto group : allData){
            for (auto pngName : group.second){
                narrowName = converter.to_bytes(pngName);

                lastPeriod = narrowName.rfind(".");
                extension = narrowName.substr(lastPeriod+1, narrowName.size()-1);

                if (extension == "obj"){
                    data3d[scName].loadObj(
                        narrowFolderPath,
                        group.first,
                        narrowName
                    );
                }

                
            }
        }

        data3d[scName].loadCam(L"deveCam");
    }
}


void Scene::release(std::string scName){
    if(names[scName] == GLPA_SCENE_2D){
        for(auto it2d : data2d){
            it2d.second.release();
        }
    }
    else if (names[scName] == GLPA_SCENE_3D){
        for(auto it3d : data3d){
            it3d.second.release();
        }
    }
}