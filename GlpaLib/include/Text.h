#ifndef GLPA_TEXT_H_
#define GLPA_TEXT_H_

#include "SceneObject.h"

#include <d2d1.h>
#include <dwrite.h>
#pragma comment(lib, "d2d1")

namespace Glpa
{

class Text : public Glpa::SceneObject
{
private :
    bool visible = true;
    Glpa::Vec2d pos;

    IDWriteFactory* pDWriteFactory = nullptr;
    IDWriteTextFormat* pTextFormat = nullptr;

public :
    Text();
    ~Text() override;

    void load() override;
    void release() override;
};

}


#endif GLPA_TEXT_H_