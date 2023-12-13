#ifndef SCENE_H_
#define SCENE_H_

#include <string>
#include <unordered_map>
#include <Windows.h>

#include <locale>
#include <codecvt>

#include "scene2d.h"
#include "scene3d.h"

#include "error.h"

#define GLPA_SCENE_2D 0
#define GLPA_SCENE_3D 1

class Scene
{
public :
    void setFolderPass(std::wstring scene_name_folder_pass);
    
    void create(std::string scene_name, int select_type);
    void load(
        std::string scene_name, 
        std::wstring folder_path, 
        std::unordered_map<std::wstring, std::vector<std::wstring>> all_data
    );
    void release(std::string scene_name);
    void reload();
    void remove();
    void update();

    

private :
    std::wstring folderPass;
    std::unordered_map<std::string, int> names;
    std::unordered_map<std::string, Scene2d> data2d;
    std::unordered_map<std::string, Scene3d> data3d;

};

#endif  SCENE_H_

