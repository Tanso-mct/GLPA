#ifndef GLPA_COLOR_H_
#define GLPA_COLOR_H_

#include <Windows.h>


namespace Glpa
{

class Color
{
private :
    BYTE sourceR, sourceG, sourceB, sourceA;
    BYTE nowR, nowG, nowB, nowA;

public :
    BYTE GetSourceR() const {return sourceR;}
    BYTE GetSourceG() const {return sourceG;}
    BYTE GetSourceB() const {return sourceB;}
    BYTE GetSourceA() const {return sourceA;}

    BYTE GetNowR() const {return nowR;}
    BYTE GetNowG() const {return nowG;}
    BYTE GetNowB() const {return nowB;}
    BYTE GetNowA() const {return nowA;}

    void SetRbga(DWORD buf);

    void AlphaBlend(Glpa::Color bg);
    
};

}

#endif GLPA_COLOR_H_

