#include "scene2d.h"

void Scene2d::loadPng(std::string folderPath, std::string groupName, std::string fileName){
    Image tempImage;
    std::string filePath = folderPath + "/" + fileName;

    std::size_t strForwardPosX = fileName.find(GLPA_SCENE2D_FILENAME_X);
    std::size_t strBehindPosX = fileName.find(GLPA_SCENE2D_FILENAME_X) + GLPA_SCENE2D_FILENAME_X_SIZE;
    std::size_t strForwardPosY = fileName.rfind(GLPA_SCENE2D_FILENAME_Y);
    std::size_t strBehindPosY = fileName.rfind(GLPA_SCENE2D_FILENAME_Y) + GLPA_SCENE2D_FILENAME_Y_SIZE;

    std::string coordinateX, coordinateY;
    if (
        strForwardPosX == std::string::npos ||
        strBehindPosX == std::string::npos ||
        strForwardPosY == std::string::npos ||
        strBehindPosY == std::string::npos
    ){
        throw std::runtime_error(ERROR_SCENE2D_LOADPNG);
    }

    coordinateX = fileName.substr(strBehindPosX, strForwardPosY - strBehindPosX);
    coordinateY = fileName.substr(strBehindPosY, fileName.size() - strBehindPosY);
    tempImage.pos.x = std::stod(coordinateX);
    tempImage.pos.y = std::stod(coordinateY);

    std::string cutFileName = fileName.substr(0, strForwardPosX);

    tempImage.png.load(filePath);

    if(group.find(groupName) != group.end()){
        group[groupName].push_back(cutFileName);
    }
    else{
        std::vector<std::string> tempVec;
        group.emplace(groupName, tempVec);
        group[groupName].push_back(cutFileName);
    }

    pngAttribute.emplace(cutFileName, tempImage);
}
