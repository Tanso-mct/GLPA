#include "scene2d.h"

void Scene2d::loadPng(std::string folderPath, std::string groupName, std::string fileName){
    Image tempImage;
    std::string filePath = folderPath + "/" + fileName;

    std::size_t strForwardPosX = fileName.find(GLPA_SCENE2D_FILENAME_X);
    std::size_t strBehindPosX = fileName.find(GLPA_SCENE2D_FILENAME_X) + GLPA_SCENE2D_FILENAME_X_SIZE;
    std::size_t strForwardPosY = fileName.rfind(GLPA_SCENE2D_FILENAME_Y);
    std::size_t strBehindPosY = fileName.rfind(GLPA_SCENE2D_FILENAME_Y) + GLPA_SCENE2D_FILENAME_Y_SIZE;

    std::size_t strForwardLayer = fileName.find(GLPA_SCENE2D_FILENAME_L);
    std::size_t strBehindLayer = fileName.find(GLPA_SCENE2D_FILENAME_L) + GLPA_SCENE2D_FILENAME_L_SIZE;

    std::string coordinateX, coordinateY;
    if (
        strForwardPosX == std::string::npos ||
        strBehindPosX == std::string::npos ||
        strForwardPosY == std::string::npos ||
        strBehindPosY == std::string::npos ||
        strForwardLayer == std::string::npos ||
        strBehindLayer == std::string::npos
    ){
        throw std::runtime_error(ERROR_SCENE2D_LOADPNG);
    }

    coordinateX = fileName.substr(strBehindPosX, strForwardPosY - strBehindPosX);
    coordinateY = fileName.substr(strBehindPosY, strForwardPosY);
    tempImage.pos.x = std::stod(coordinateX);
    tempImage.pos.y = std::stod(coordinateY);

    std::string cutFileName = fileName.substr(0, strForwardPosX);

    if(layerOrder.find(groupOrder[groupName]) != layerOrder.end()){
        layerOrder[groupOrder[groupName]].emplace(
            std::stod(fileName.substr(strBehindLayer, fileName.size() - strBehindLayer)),
            cutFileName
        );
    }
    else{
        std::unordered_map<int, std::string> temp;
        temp.emplace(
            std::stod(fileName.substr(strBehindLayer, fileName.size() - strBehindLayer)),
            cutFileName
        );
        layerOrder.emplace(groupOrder[groupName], temp);
    }

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


void Scene2d::loadText(){
    text.addGroup(
        L"Temp",
        24,
        GLPA_SYSTEM_FIXED_FONT,
        {204, 204, 204},
        FALSE,
        {10, 10},
        {1000, 650},
        true
    );

    text.addText(L"Temp", L"/glpa temp");
    text.addText(L"Temp", L"/glpa temp2");
    text.addText(L"Temp", L"/glpa temp3");
}


void Scene2d::release(){
    for(auto it : pngAttribute){
        it.second.png.release();
    }

    text.releaseAllGroup();

    pngAttribute.clear();
}


void Scene2d::edit(HDC hBufDC, LPDWORD lpPixel, int width, int height, int dpi){
    for (auto it : sceneFrameFunc){
        it.second(hBufDC, lpPixel, width, height, dpi);
    }
}


void Scene2d::update(HDC hBufDC, LPDWORD wndBuffer, int wndWidth, int wndHeight, int wndDpi){
    if (edited){
        for(int y = 0; y < wndHeight; y++)
        {
            for(int x = 0; x < wndWidth; x++)
            {
                wndBuffer[(x+y*wndWidth * wndDpi)] = 0x00000000;
            }
        }

        int drawPoint;
        BYTE alpha, backAlpha;
        Image it;
        for (int i = 1; i <= layerOrder.size(); i++){
            for (int j = 1; j <= layerOrder[i].size(); j++){
                it = pngAttribute[layerOrder[i][j]];
                if (it.visible){
                    drawPoint = it.pos.x + it.pos.y*wndWidth * wndDpi;
                    for(int y = 0; y < it.png.height; y++)
                    {
                        for(int x = 0; x < it.png.width; x++)
                        {
                            if(
                                (it.pos.x + x) >= 0 && (it.pos.y + y) >= 0 &&
                                (it.pos.x + x) < wndWidth && (it.pos.y + y) < wndHeight
                            ){
                                alpha = (it.png.data[x+y*it.png.width] >> 24) & 0xFF;
                                backAlpha = wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)];

                                if (alpha != 0x00){
                                    if (backAlpha == 0x00){
                                        wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)]
                                        = color.alphaBlend(
                                            (DWORD)(it.png.data[x+y*it.png.width]), 
                                            0xFF000000
                                        );
                                    }
                                    else{
                                        wndBuffer[drawPoint + (x+y*wndWidth * wndDpi)]
                                        = color.alphaBlend(
                                            (DWORD)(it.png.data[x+y*it.png.width]), 
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

        

        text.drawAll(hBufDC);
        
        edited = false;
    }
}


void Scene2d::addSceneFrameFunc(std::wstring funcName, GLPA_SCENE_FUNC_FUNCTIONAL addFunc){
    sceneFrameFunc[funcName] = addFunc;
}


void Scene2d::editSceneFrameFunc(std::wstring funcName, GLPA_SCENE_FUNC_FUNCTIONAL editedFunc){
    if (sceneFrameFunc.find(funcName) != sceneFrameFunc.end()){
        sceneFrameFunc[funcName] = editedFunc;
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_2D_FRAME_FUNC_NAME);
    }
    
}


void Scene2d::releaseSceneFrameFunc(std::wstring funcName){
    if (sceneFrameFunc.find(funcName) != sceneFrameFunc.end()){
        sceneFrameFunc.erase(funcName);
    }
    else{
        throw std::runtime_error(ERROR_GLPA_SCENE_2D_FRAME_FUNC_NAME);
    }
}
