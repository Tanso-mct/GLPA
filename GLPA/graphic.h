#ifndef GRAPHIC_H_
#define GRAPHIC_H_

#include <tchar.h>
#include <windows.h>
#include <vector>
#include <string>

#include "cgmath.cuh"

//For LAU
extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];
extern POINT pt;

//For PLAY
extern TCHAR szstrfPlay[256];
extern TCHAR mouseMsgfPlay[256];
extern POINT ptfPlay;

typedef struct tagNUMCOMB3
{
    int num1; // number 1
    int num2; // number 2
    int num3; // number 3
} NUMCOMB3;

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
    int format;
    std::vector<RGBA> rgbaData;
} IMAGE;

typedef struct tagVERTEX
{
    std::vector<VECTOR3D> world; // world coordinate
    std::vector<VECTOR2D> uv; // uv coordinate
    std::vector<VECTOR3D> normal; // normal
} VERTEX;

typedef struct tagPOLYGON
{
    std::vector<NUMCOMB3> v; // vertex number
    std::vector<NUMCOMB3> uv; // uv number
    std::vector<NUMCOMB3> normal; // normal number
} POLYGON;

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
