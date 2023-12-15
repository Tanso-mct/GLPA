#ifndef COLOR_H_
#define COLOR_H_

#include <Windows.h>

#define GLPA_NEW_COLOR 0
#define GLPA_BACK_COLOR 1

typedef struct tagRGB{
    BYTE r, g, b;
} Rgb;

typedef struct tagRGBA{
    BYTE r, g, b, a;
} Rgba;


class Color
{
public :  
    void getRgba(DWORD buffer, int color_type);
    void setRgba();

    DWORD alphaBlend(DWORD new_buffer, DWORD back_buffer);
    
private :
    Rgba newColor;
    Rgba backColor;
    Rgba resultColor;
    DWORD resultBuffer;
};

#endif COLOR_H_


