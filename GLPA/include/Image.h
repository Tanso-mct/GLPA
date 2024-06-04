#ifndef GLPA_IMAGE_H_
#define GLPA_IMAGE_H_

#include "Png.h"

namespace Glpa
{

class Image : public Glpa::Png
{
private :
    bool visible = true;
    int drawOrder = 0;

    Glpa::Vec2d pos;

public :
    Image(std::string argName, std::string filePath, Glpa::Vec2d defPos);
    ~Image() override;

    bool getVisible() const {return visible;}
    void setVisible(bool symbol) {visible = symbol;}

    int getDrawOrder() const {return drawOrder;}
    void setDrawOrder(int value) {drawOrder = value;}

    Glpa::Vec2d getPos() const {return pos;}
    void setPos(Glpa::Vec2d value) {pos = value;}

};

}

#endif GLPA_IMAGE_H_
