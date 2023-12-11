#include "scene2d.h"

void Scene2d::loadPng(std::string folderPath, std::string groupName, std::string fileName){
    Image tempImage;
    std::string filePath = folderPath + "/" + fileName;
    tempImage.png.load(filePath);

    if(group.find(groupName) != group.end()){
        group[groupName].push_back(fileName);
    }
    else{
        std::vector<std::string> tempVec;
        group.emplace(groupName, tempVec);
        group[groupName].push_back(fileName);
    }

    pngAttribute.emplace(fileName, tempImage);
}
