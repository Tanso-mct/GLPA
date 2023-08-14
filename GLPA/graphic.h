#ifndef GRAPHIC_H_
#define GRAPHIC_H_

#include <tchar.h>
#include <windows.h>
#include <vector>

#include "window.h"

#define FILETYPE_BMP 0
#define FILETYPE_PNG 1

//For LAU
extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];
extern POINT pt;

//For PLAY
extern TCHAR szstrfPlay[256];
extern TCHAR mouseMsgfPlay[256];
extern POINT ptfPlay;

typedef struct tagRGBA
{
    int r;
    int g;
    int b;
    int a;
} RGBA;

typedef struct tagIMAGE
{
    int width;
    int height;
    int colorDepth;
    int compType;
    std::vector<RGBA> data;
    int format;
} IMAGE;

typedef struct tagVECTOR2
{
    double x;
    double y;
} VEC2;

typedef struct tagVECTOR3
{
    double x;
    double y;
    double z;
} VEC3;

class SCREEN
{
    public :
};

class SCR_LAU : public SCREEN
{
    public :
    int scrUpdate();
};

class SCR_PLAY : public SCREEN
{
    public :
    int scrUpdate();
};


// Function to change the value of what is drawn on the screen
// TODO: change class
void scrLAUDwgContModif(HDC hBuffer_DC/*, TEXTURE *texture*/);
void scrPLAYDwgContModif(HDC hBuffer_DC/*, TEXTURE *texture*/);

#endif GRAPHIC_H_
