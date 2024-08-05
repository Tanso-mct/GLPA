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
    Glpa::Vec2d size = {300, 300};

    std::wstring fontName = L"Arial";
    DWRITE_FONT_WEIGHT fontWeight = DWRITE_FONT_WEIGHT_NORMAL;
    DWRITE_FONT_STYLE fontStyle = DWRITE_FONT_STYLE_NORMAL;
    DWRITE_FONT_STRETCH fontStretch = DWRITE_FONT_STRETCH_NORMAL;
    float fontSize = 32;
    DWRITE_READING_DIRECTION readingDirect = DWRITE_READING_DIRECTION_LEFT_TO_RIGHT;
    float lineSpacing = 40;
    float baselineOffSet = 32;
    std::wstring localeName = L"en-us";

    std::wstring words = L"New text";

    IDWriteFactory* pDWriteFactory = nullptr;
    IDWriteTextLayout* pTextLayout = nullptr;
    IDWriteTextFormat* pTextFormat = nullptr;

    UINT32 lineCount = 0;
    std::vector<DWRITE_LINE_METRICS> lineMetrics;

    ID2D1SolidColorBrush* pBrush = nullptr;
    D2D1::ColorF brushColor = D2D1::ColorF::White;

public :
    Text(std::string argName);
    ~Text() override;

    void EditPos(Glpa::Vec2d argPos){pos = argPos;}
    void EditSize(Glpa::Vec2d argSize){size = argSize;}
    void EditFontName(std::string argFontName);
    void EditFontWeight(DWRITE_FONT_WEIGHT argFontWeight){fontWeight = argFontWeight;}
    void EditFontStyle(DWRITE_FONT_STYLE argFontStyle){fontStyle = argFontStyle;}
    void EditFontStretch(DWRITE_FONT_STRETCH argFontStretch){fontStretch = argFontStretch;}
    void EditFontSize(float argFontSize){fontSize = argFontSize;}
    void EditReadingDirect(DWRITE_READING_DIRECTION argReadingDirect){readingDirect = argReadingDirect;}
    void EditLineSpacing(float argLineSpacing){lineSpacing = argLineSpacing;}
    void EditBaselineOffSet(float argBaselineOffSet){baselineOffSet = argBaselineOffSet;}
    void EditLocaleName(std::string argLocaleName);

    void EditWords(std::string argWords);
    void EditColor(UINT32 argColor){brushColor = D2D1::ColorF(argColor);}

    Glpa::Vec2d GetPos() const {return pos;}
    Glpa::Vec2d GetSize() const {return size;}
    std::string GetFontName();
    DWRITE_FONT_WEIGHT GetFontWeight() const {return fontWeight;}
    DWRITE_FONT_STYLE GetFontStyle() const {return fontStyle;}
    DWRITE_FONT_STRETCH GetFontStretch() const {return fontStretch;}
    float GetFontSize() const {return fontSize;}
    DWRITE_READING_DIRECTION GetReadingDirect() const {return readingDirect;}
    float GetLineSpacing() const {return lineSpacing;}
    float GetBaselineOffSet() const {return baselineOffSet;}
    std::string GetLocaleName();

    std::string GetWords();
    D2D1::ColorF GetColor() const {return brushColor;}

    void load() override;
    void release() override;

    void drawText(ID2D1HwndRenderTarget* pRenderTarget);
};

}


#endif GLPA_TEXT_H_