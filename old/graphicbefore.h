#ifndef GRAPHIC_H_
#define GRAPHIC_H_

#include <tchar.h>
#include <windows.h>
#include <vector>
#include <string>

//For LAU
extern TCHAR szstr[256];
extern TCHAR mouseMsg[256];
extern POINT pt;

//For PLAY
extern TCHAR szstrfPlay[256];
extern TCHAR mouseMsgfPlay[256];
extern POINT ptfPlay;

// TODO: Create a SCREEN file
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

