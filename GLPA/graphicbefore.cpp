#include "graphic.h"

#include "windowbefore.h"

//For LAU
TCHAR szstr[256] = _T("NAN KEY");
TCHAR mouseMsg[256] = _T("NOW COORDINATE");
POINT pt = 
{
    5,
    20
};

//For PLAY
TCHAR szstrfPlay[256] = _T("NAN KEY");
TCHAR mouseMsgfPlay[256] = _T("NOW COORDINATE");
POINT ptfPlay = 
{
    5,
    20
};

void scrLAUDwgContModif(HDC hBuffer_DC/*, TEXTURE *texture*/)
{
    // texture->displayImage_rectangle(
    //     WND_LAU.lpPixel, texture->file1.bmp_pixel, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, 
    //     0, 0,
    //     0, 0,
    //     FILE_MAXPIXEL_X, FILE_MAXPIXEL_Y
    // );

    HFONT hFont1 = CreateFont(30 * DISPLAY_RESOLUTION, 0, 
		0, 0, 0, 
		FALSE, FALSE, FALSE,   
		SHIFTJIS_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,  
		VARIABLE_PITCH | FF_ROMAN, NULL);   
	SelectObject(hBuffer_DC, hFont1);  
    
    TextOut(
        hBuffer_DC,
        pt.x,
        pt.y,
        mouseMsg,
        _tcslen(mouseMsg)
    );
    DeleteObject(hFont1); 

    TextOut(
        hBuffer_DC,
        5,
        5,
        szstr,
        _tcslen(szstr)
    );
}

void scrPLAYDwgContModif(HDC hBuffer_DC/*, TEXTURE *texture*/)
{
    // texture->displayImage_rectangle(
    //     WND_LAU.lpPixel, texture->file1.bmp_pixel, WINDOW_WIDTH, WINDOW_HEIGHT, DISPLAY_RESOLUTION, 
    //     0, 0,
    //     0, 0,
    //     FILE_MAXPIXEL_X, FILE_MAXPIXEL_Y
    // );

    HFONT hFont1 = CreateFont(30 * DISPLAY_RESOLUTION, 0, 
		0, 0, 0, 
		FALSE, FALSE, FALSE,   
		SHIFTJIS_CHARSET, OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,  
		VARIABLE_PITCH | FF_ROMAN, NULL);   
	SelectObject(hBuffer_DC, hFont1);  
    
    TextOut(
        hBuffer_DC,
        ptfPlay.x,
        ptfPlay.y,
        mouseMsgfPlay,
        _tcslen(mouseMsgfPlay)
    );
    DeleteObject(hFont1); 

    TextOut(
        hBuffer_DC,
        5,
        5,
        szstrfPlay,
        _tcslen(szstrfPlay)
    );
}