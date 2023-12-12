#include "scene2d.h"

void Scene2d::loadPng(std::string folderPath, std::string groupName, std::string fileName){
    Image tempImage;
    std::string filePath = folderPath + "/" + fileName;

    std::size_t strFirstSpace = fileName.find(" ");
    std::size_t strSecondSpace = fileName.rfind(" ");

    std::string coordinateX, coordinateY;
    if (strFirstSpace != std::string::npos || strSecondSpace != std::string::npos){
        throw std::runtime_error()
    }

    coordinateX = fileName.substr(strFirstSpace, strSecondSpace - strFirstSpace);
    coordinateY = fileName.substr(strSecondSpace, fileName.size() - strSecondSpace);
    tempImage.pos.x = std::stod(coordinateX);
    tempImage.pos.y = std::stod(coordinateY);

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
