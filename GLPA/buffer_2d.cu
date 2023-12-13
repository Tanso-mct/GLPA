#include "buffer_2d.cuh"

void Buffer2d::GetCalcColorComponents(DWORD rgbaValue) {
    red = (rgbaValue & 0xFF000000) >> 24;
    green = (rgbaValue & 0x00FF0000) >> 16;
    blue = (rgbaValue & 0x0000FF00) >> 8;
    alpha = rgbaValue & 0x000000FF; 

    red = red / 255;
    green = green / 255;
    blue = blue / 255;
    alpha = alpha / 255;
}


void Buffer2d::GetBackColorComponents(DWORD rgbaValue){
    backRed = (rgbaValue & 0xFF000000) >> 24;
    backGreen = (rgbaValue & 0x00FF0000) >> 16;
    backBlue = (rgbaValue & 0x0000FF00) >> 8;
    backAlpha = rgbaValue & 0x000000FF;

    backRed = backRed / 255;
    backGreen = backGreen / 255;
    backBlue = backBlue / 255;
    backAlpha = backAlpha / 255;
}


void Buffer2d::SetRGBAValue(DWORD* rgbaValue){

    *rgbaValue |= (DWORD)resultRed << 24;
    *rgbaValue |= (DWORD)resultGreen << 16;
    *rgbaValue |= (DWORD)resultBlue << 8;
    *rgbaValue |= (DWORD)alpha;
}

DWORD Buffer2d::alphaBlend(DWORD newColor, DWORD backColor){
    GetCalcColorComponents(newColor);
    GetBackColorComponents(backColor);

    // final color = background color + (overlap color - background color) * (alpha / 255)
    resultRed = (backRed * (1 - alpha) + red * alpha) * 255;
    resultGreen = (backGreen * (1 - alpha) + green * alpha) * 255;
    resultBlue = (backBlue * (1 - alpha) + blue * alpha) * 255;

    DWORD rtRgbaValue;
    SetRGBAValue(&rtRgbaValue);

    return rtRgbaValue;
}
