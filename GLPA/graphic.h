#ifndef GRAPHIC_H_
#define GRAPHIC_H_

#include <Windows.h>

//TODO:add prefix "gr" to the name of the global variable
//LAUNCHER Win32 Define
PAINTSTRUCT hPS;
HDC hWindow_DC;

//LAUNCHER Buffer DC
HDC hBuffer_DC;
HBITMAP hBuffer_bitmap;    
BITMAPINFO hBuffer_bitmapInfo; 

//TODO:textureÇç\ë¢ëÃÇ…ïœçXÇ∑ÇÈ
//Texture
TEXTURE texture_sample;
TEXTURE *pt_texture_sample = &texture_sample;

//bmpfile
BMPFILE sample;    
BMPFILE *pt_sample = &sample;

BMPFILE sample2;    
BMPFILE *pt_sample2 = &sample2;

BMPFILE sample3;    
BMPFILE *pt_sample3 = &sample3;

//fps
int refreshRate;
bool startFpsCount = false;
clock_t thisloop;
clock_t lastloop;
long double fps;

#endif GRAPHIC_H_
