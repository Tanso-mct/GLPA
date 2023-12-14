#include "color.h"

void Color::getRgba(DWORD buffer, int colorType){
    if (colorType == GLPA_NEW_COLOR){
        newColor.r = (buffer >> 16) & 0xFF;
        newColor.g = (buffer >> 8) & 0xFF;
        newColor.b = buffer & 0xFF;
        newColor.a = (buffer >> 24) & 0xFF;
    }
    else if (colorType == GLPA_BACK_COLOR){
        backColor.r = (buffer >> 16) & 0xFF;
        backColor.g = (buffer >> 8) & 0xFF;
        backColor.b = buffer & 0xFF;
        backColor.a = (buffer >> 24) & 0xFF;
    }
}

void Color::setRgba(){
    resultBuffer = (resultColor.a << 24) | (resultColor.r << 16) | (resultColor.g << 8) | resultColor.b;
}

DWORD Color::alphaBlend(DWORD newBuffer, DWORD backBuffer){
    getRgba(newBuffer, GLPA_NEW_COLOR);
    getRgba(backBuffer, GLPA_BACK_COLOR);

    float alpha = static_cast<float>(newColor.a) / 255.0f;
    float invAlpha = 1.0f - alpha;

    resultColor.r = static_cast<unsigned char>(alpha * newColor.r + invAlpha * backColor.r);
    resultColor.g = static_cast<unsigned char>(alpha * newColor.g + invAlpha * backColor.g);
    resultColor.b = static_cast<unsigned char>(alpha * newColor.b + invAlpha * backColor.b);
    resultColor.a = static_cast<unsigned char>(newColor.a + invAlpha * backColor.a);

    setRgba();

    return resultBuffer;
}