#include "Text.h"

Glpa::Text::Text(std::string argName)
{
    name = argName;
    type = Glpa::CLASS_TEXT;
}

Glpa::Text::~Text()
{
}

void Glpa::Text::EditFontName(std::string argFontName)
{
    fontName = strConverter.from_bytes(argFontName);
}

void Glpa::Text::EditLocaleName(std::string argLocaleName)
{
    localeName = strConverter.from_bytes(argLocaleName);
}

void Glpa::Text::EditWords(std::string argWords)
{
    words = strConverter.from_bytes(argWords);
}

std::string Glpa::Text::GetFontName()
{
    return strConverter.to_bytes(fontName);
}

std::string Glpa::Text::GetLocaleName()
{
    return strConverter.to_bytes(localeName);
}

std::string Glpa::Text::GetWords()
{
    return strConverter.to_bytes(words);
}

int Glpa::Text::GetLineTextCount(int line)
{
    if (pTextLayout != nullptr) {
        pTextLayout->Release();
    }

    pDWriteFactory->CreateTextLayout(
        words.c_str(),
        (UINT32)words.length(),
        pTextFormat,
        size.x, size.y,
        &pTextLayout
    );

    UINT32 lineCount = 0;
    pTextLayout->GetLineMetrics(nullptr, 0, &lineCount);
    lineMetrics.clear();
    lineMetrics.resize(lineCount);
    pTextLayout->GetLineMetrics(lineMetrics.data(), lineCount, &lineCount);

    return lineMetrics[line].length;
}

int Glpa::Text::GetLineCount()
{
    if (pTextLayout != nullptr) {
        pTextLayout->Release();
    }

    pDWriteFactory->CreateTextLayout(
        words.c_str(),
        (UINT32)words.length(),
        pTextFormat,
        size.x, size.y,
        &pTextLayout
    );

    UINT32 lineCount = 0;
    pTextLayout->GetLineMetrics(nullptr, 0, &lineCount);
    lineMetrics.clear();
    lineMetrics.resize(lineCount);
    pTextLayout->GetLineMetrics(lineMetrics.data(), lineCount, &lineCount);

    return lineCount;
}

void Glpa::Text::load()
{
    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&pDWriteFactory));

    HRESULT hr = pDWriteFactory->CreateTextFormat
    (
        fontName.c_str(), nullptr, fontWeight, fontStyle, fontStretch, fontSize, localeName.c_str(), &pTextFormat
    );

    pTextFormat->SetReadingDirection(readingDirect);
    pTextFormat->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, lineSpacing, baselineOffSet);

    if (pTextLayout != nullptr) {
        pTextLayout->Release();
    }

    pDWriteFactory->CreateTextLayout(
        words.c_str(),
        (UINT32)words.length(),
        pTextFormat,
        size.x, size.y,
        &pTextLayout
    );

    UINT32 lineCount = 0;
    pTextLayout->GetLineMetrics(nullptr, 0, &lineCount);
    lineMetrics.resize(lineCount);
    pTextLayout->GetLineMetrics(lineMetrics.data(), lineCount, &lineCount);
}

void Glpa::Text::release()
{
    pTextFormat->Release();
    pDWriteFactory->Release();
}

void Glpa::Text::drawText(ID2D1HwndRenderTarget *pRenderTarget)
{
    pRenderTarget->CreateSolidColorBrush(brushColor, &pBrush);

    pRenderTarget->DrawText
    (
        words.c_str(), words.size(), pTextFormat, D2D1::RectF(size.x, size.y, pos.x, pos.y), pBrush
    );

    pBrush->Release();
}
