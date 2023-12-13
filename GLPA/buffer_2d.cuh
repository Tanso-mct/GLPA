#ifndef BUFFER2D_CUH_
#define BUFFER2D_CUH_

#include <Windows.h>

class Buffer2d
{
public :  
    void GetCalcColorComponents(DWORD rgba_value);
    void GetBackColorComponents(DWORD rgba_value);

    void SetRGBAValue(DWORD* rgba_value);

    DWORD alphaBlend(DWORD new_color, DWORD back_color);
    
private :
    double red;
    double green;
    double blue;
    double alpha;

    double backRed;
    double backGreen;
    double backBlue;
    double backAlpha;

    int resultRed;
    int resultGreen;
    int resultBlue;
    int resultAlpha;
};

#endif  BUFFER2D_CUH_


