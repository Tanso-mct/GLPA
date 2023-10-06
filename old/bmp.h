
#ifndef BMP_H_
#define BMP_H_

#include <tchar.h>
#include <windows.h>

#define FILE_MAXPIXEL_X 1920
#define FILE_MAXPIXEL_Y 1080

class BMPFILE
{
    public :
    HDC hImageDC;
    HBITMAP hbitmapImage;
    BITMAP bitmapImage;

    int pixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y];
        
    bool deleteImage();
    bool load(LPCTSTR link, HDC hdc);
    bool create(
        unsigned short windowWidth,
        unsigned short windowHeight,
        unsigned short displayResolution,
        HDC hBmpDC,
        LPDWORD bmpPixel    
    );

    int getWidth();
    int getHeight();

    int pixelDecColor(int pixelX, int pixelY);
};

class IMAGE_BMP
{
    public :
    UINT bmp_pixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y];
    UINT width;
    UINT height;
    bool load(int imagePixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y], int imageWidth, int imageHeight);
};

class TEXTURE
{
    public :
    int loaded = 0;
    IMAGE_BMP file1;
    IMAGE_BMP file2;
    IMAGE_BMP file3;
    bool insertBMP(int pixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y], int width, int height);
    int pixelDecColor(int file_bmp[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y] ,int pixelX, int pixelY);
    bool displayImage_rectangle
    (
        LPDWORD lpPixel,
        UINT bmp_pixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y],
        UINT windowWidth,
        UINT windowHeight,
        UINT displayResolution,
        UINT windowStartX,
        UINT windowStartY,
        UINT imageStartX,
        UINT imageStartY,
        UINT imageEndX,
        UINT imageEndY
    );
};

// Texture
extern TEXTURE texture_sample;
extern TEXTURE *pt_texture_sample;

// bmpfile
extern BMPFILE sample;    
extern BMPFILE *pt_sample;

extern BMPFILE sample2;    
extern BMPFILE *pt_sample2;

extern BMPFILE sample3;    
extern BMPFILE *pt_sample3;

#endif // BMP_H_
