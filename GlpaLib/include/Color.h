#ifndef GLPA_COLOR_H_
#define GLPA_COLOR_H_

#include <Windows.h>


namespace Glpa
{

class Color
{
private :
    BYTE sourceR = 0;
    BYTE sourceG = 0;
    BYTE sourceB = 0;
    BYTE sourceA = 0;

    BYTE nowR = 0;
    BYTE nowG = 0;
    BYTE nowB = 0;
    BYTE nowA = 0;
    
public :
    Color(BYTE r, BYTE g, BYTE b, BYTE a);
    Color(DWORD buf);
    Color(){}

    BYTE GetSourceR() const {return sourceR;}
    BYTE GetSourceG() const {return sourceG;}
    BYTE GetSourceB() const {return sourceB;}
    BYTE GetSourceA() const {return sourceA;}

    BYTE GetNowR() const {return nowR;}
    BYTE GetNowG() const {return nowG;}
    BYTE GetNowB() const {return nowB;}
    BYTE GetNowA() const {return nowA;}

    void SetRgba(DWORD buf);

    DWORD GetDword();

    void AlphaBlend(Glpa::Color bg);
    
};

}

#endif GLPA_COLOR_H_

