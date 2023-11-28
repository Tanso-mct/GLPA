#ifndef GLPA_H_
#define GLPA_H_

#include "window_api.h"

class Glpa
{
public :
    void createWindow();

    void showWindow();
    
    void updateWindowInfo();

    void deleteWindow();

    void graphicLoop();

    void createScene();

    void loadScene();

    void setSceneUserInputFunc();

    void setSceneActionFunc();

    void setSceneFrameFunc();

    void selectUseScene();

    void selectUseCamera();

    void inputCameraInfo();

    void inputObjectInfo();

    void inputCharacterInfo();

};

#endif  GLPA_H_
