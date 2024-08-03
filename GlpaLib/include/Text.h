#ifndef GLPA_TEXT_H_
#define GLPA_TEXT_H_

#include "SceneObject.h"

#include <d2d1.h>
#include <dwrite.h>
#pragma comment(lib, "d2d1")
#pragma comment(lib, "dwrite")

namespace Glpa
{

class Text : public Glpa::SceneObject
{
private :
    bool visible = true;
    Glpa::Vec2d pos = {0, 0};

    std::wstring fontName = L"Arial";
    DWRITE_FONT_WEIGHT fontWeight = DWRITE_FONT_WEIGHT_NORMAL;
    DWRITE_FONT_STYLE fontStyle = DWRITE_FONT_STYLE_NORMAL;
    DWRITE_FONT_STRETCH fontStretch = DWRITE_FONT_STRETCH_NORMAL;
    float fontSize = 32;
    std::wstring localeName = L"en-us";

    std::wstring words = L"New text";

    IDWriteFactory* pDWriteFactory = nullptr;
    IDWriteTextFormat* pTextFormat = nullptr;

    ID2D1SolidColorBrush* pBrush = nullptr;
    D2D1::ColorF brushColor = D2D1::ColorF::Black;

public :
    Text();
    ~Text() override;

    void EditPos(Glpa::Vec2d argPos){pos = argPos;}
    void EditFontName(std::string argFontName);
    void EditFontWeight(DWRITE_FONT_WEIGHT argFontWeight){fontWeight = argFontWeight;}
    void EditFontStyle(DWRITE_FONT_STYLE argFontStyle){fontStyle = argFontStyle;}
    void EditFontStretch(DWRITE_FONT_STRETCH argFontStretch){fontStretch = argFontStretch;}
    void EditFontSize(float argFontSize){fontSize = argFontSize;}
    void EditLocaleName(std::string argLocaleName);

    void EditWords(std::string argWords);
    void EditColor(UINT32 argColor){brushColor = D2D1::ColorF(argColor);}

    Glpa::Vec2d GetPos(){return pos;}
    std::string GetFontName();
    DWRITE_FONT_WEIGHT GetFontWeight(){return fontWeight;}
    DWRITE_FONT_STYLE GetFontStyle(){return fontStyle;}
    DWRITE_FONT_STRETCH GetFontStretch(){return fontStretch;}
    float GetFontSize(){return fontSize;}
    std::string GetLocaleName();

    std::string GetWords();
    D2D1::ColorF GetColor(){return brushColor;}

    void load() override;
    void release() override;

    void drawText(ID2D1HwndRenderTarget* pRenderTarget);
};

}


#endif GLPA_TEXT_H_