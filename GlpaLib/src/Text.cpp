#include "Text.h"

Glpa::Text::~Text()
{
}

void Glpa::Text::load()
{
    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&pDWriteFactory));

    pDWriteFactory->CreateTextFormat
    (
        fontName.c_str(), nullptr, fontWeight, fontStyle, fontStretch, fontSize, localeName.c_str(), &pTextFormat
    );
}

void Glpa::Text::release()
{
    pTextFormat->Release();
    pBrush->Release();
    pDWriteFactory->Release();
}
