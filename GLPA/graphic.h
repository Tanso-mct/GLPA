#ifndef GRAPHIC_H_
#define GRAPHIC_H_

#include <tchar.h>
#include <windows.h>
#include <vector>
#include <string>

#include "window.h"

//For LAU
extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];
extern POINT pt;

//For PLAY
extern TCHAR szstrfPlay[256];
extern TCHAR mouseMsgfPlay[256];
extern POINT ptfPlay;

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

typedef struct tagNUMCOMB3
{
    unsigned short num1; // number 1
    unsigned short num2; // number 2
    unsigned short num3; // number 3
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
    std::vector<VEC3> world; // world coordinate
    std::vector<VEC2> uv; // uv coordinate
    std::vector<VEC3> normal; // normal
} VERTEX;

typedef struct tagPOLYGON
{

    std::vector<NUMCOMB3> vertexNum; // vertex number
    std::vector<NUMCOMB3> uvNum; // uv number
    std::vector<NUMCOMB3> normalNum; // normal number
} POLYGON;

typedef struct tagRANGE_CUBE
{
    VEC3 origin;
    VEC3 opposite;
} RANGE_CUBE;


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

#include "file.h"

#endif GRAPHIC_H_
