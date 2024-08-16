#ifndef GLPA_IMAGE_H_
#define GLPA_IMAGE_H_

#include "SceneObject.h"

namespace Glpa
{

class Image : public Glpa::SceneObject
{
private :
    bool visible = true;
    int drawOrder = 0;
    Glpa::Vec2d pos;

public :
    Image(std::string argName, std::string argFilePath, Glpa::Vec2d defPos);
    ~Image() override;

    void load() override;
    void release() override;

    int GetWidth(){return fileDataManager->getWidth(filePath);}
    int GetHeight(){return fileDataManager->getHeight(filePath);}
    LPDWORD GetData(){return fileDataManager->getData(filePath);}

    bool getVisible() const {return visible;}
    void setVisible(bool symbol) {visible = symbol;}

    int GetDrawOrder() const {return drawOrder;}
    void SetDrawOrder(int value) {drawOrder = value;}

    Glpa::Vec2d GetPos() const {return pos;}
    void SetPos(Glpa::Vec2d value) {pos = value;}

};

}

#endif GLPA_IMAGE_H_
