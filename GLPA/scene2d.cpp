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


void Scene2d::release(){
    for(auto it : pngAttribute){
        it.second.png.release();
    }

    pngAttribute.clear();
}


void Scene2d::update(LPDWORD wndBuffer, int wndWidth, int wndHeight, int wndDpi){
    for(int y = 0; y < wndHeight; y++)
    {
        for(int x = 0; x < wndWidth; x++)
        {
            wndBuffer[(x+y*wndWidth * wndDpi)] = 0x00000000;
        }
    }

    int drawPoint;
    for(auto it : pngAttribute){
        if (it.second.visible){
            drawPoint = it.second.pos.x + it.second.pos.y*wndWidth * wndDpi;
            BYTE alpha, backAlpha;
            for(int y = 0; y < it.second.png.height; y++)
            {
                for(int x = 0; x < it.second.png.width; x++)
                {
                    if(
                        (it.second.pos.x + x) >= 0 && (it.second.pos.y + y) >= 0 &&
                        (it.second.pos.x + x) < wndWidth && (it.second.pos.y + y) < wndHeight
                    ){
                        alpha = (it.second.png.data[x+y*it.second.png.width] >> 24) & 0xFF;
                        backAlpha = wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)];

                        if (alpha != 0x00){
                            if (backAlpha == 0x00){
                                wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)]
                                = color.alphaBlend(
                                    (DWORD)(it.second.png.data[x+y*it.second.png.width]), 
                                    0xFF000000
                                );
                            }
                            else{
                                wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)]
                                = color.alphaBlend(
                                    (DWORD)(it.second.png.data[x+y*it.second.png.width]), 
                                    (DWORD)(wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)])
                                );
                            }   
                        }
                    }
                }
            }
        }
    }
}
