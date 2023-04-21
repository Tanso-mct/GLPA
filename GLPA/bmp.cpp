
#include "bmp.h"

bool BMPFILE::deleteImage()
{
    DeleteDC(hImageDC);
    return 0;
}

bool BMPFILE::load(LPCTSTR link, HDC hdc)
{
    //load bmp data
    hbitmapImage = static_cast<HBITMAP>(LoadImage(
        NULL, 
        link,
        IMAGE_BITMAP, 
        0, 
        0,
        LR_LOADFROMFILE
    ));
    
    //check
    if(hbitmapImage==NULL){
        OutputDebugStringW(_T("image NULL\n"));
        return 1;
    }

    //bitmap image width&height get
    GetObject(hbitmapImage,sizeof(BITMAP),&bitmapImage);

    //HDC create
    hImageDC = CreateCompatibleDC(hdc);

    //loadimage -> HDC
    SelectObject(hImageDC,hbitmapImage);

    //Delete load image bmp data
    DeleteObject(hbitmapImage);
    return 0;
}

bool BMPFILE::create(
    unsigned short windowWidth,
    unsigned short windowHeight,
    unsigned short displayResolution,
    HDC hBmpDC,
    LPDWORD bmpPixel
)
{
    PatBlt(hBmpDC, 0, 0, windowWidth * displayResolution, windowHeight * displayResolution, WHITENESS);
    BitBlt(hBmpDC,0,0,bitmapImage.bmWidth,bitmapImage.bmHeight,hImageDC,0,0,SRCCOPY);
    for (int y = 0; y < bitmapImage.bmHeight; y++)
    {
        for (int x = 0; x < bitmapImage.bmWidth; x++)
        {
            pixel[x][y] = bmpPixel[x + y *windowWidth * displayResolution];
        }
    }
    return 0;
}

int BMPFILE::getWidth()
{
    return bitmapImage.bmWidth;
}

int BMPFILE::getHeight()
{
    return bitmapImage.bmHeight;
}

bool IMAGE_BMP::load(int imagePixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y], int imageWidth, int imageHeight)
{
    width = imageWidth;
    height = imageHeight;
    for (int y = 0; y < FILE_MAXPIXEL_Y; y++)
    {
        for (int x = 0; x < FILE_MAXPIXEL_X; x++)
        {
            bmp_pixel[x][y] = imagePixel[x][y];
        }
    }
    return 0;
}

int TEXTURE::pixelDecColor(int file_bmp[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y] ,int pixelX, int pixelY)
{
    return file_bmp[pixelX - 1][pixelY - 1];
}

bool TEXTURE::insertBMP(int pixel[FILE_MAXPIXEL_X][FILE_MAXPIXEL_Y], int width, int height)
{
    if (loaded == 0)
    {
        file1.load(pixel, width, height);
        loaded += 1;
        return 0;
    } else if (loaded == 1)
    {
        file2.load(pixel, width, height);
        loaded += 1;
        return 0;
    } else if (loaded == 2)
    {
        file3.load(pixel, width, height);
        loaded += 1;
        return 0;
    }
    return 0;
}

bool TEXTURE::displayImage_rectangle
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
)
{
    for(UINT y = windowStartY; y <= windowStartY + (imageEndY - imageStartY) - 1; y++)
    {
        for(UINT x = windowStartX; x <= windowStartX + (imageEndX - imageStartX) - 1; x++)
        {
            if (x < windowWidth && y < windowHeight)
            {
                lpPixel[x+y*windowWidth * displayResolution] 
                = bmp_pixel[imageStartX + (x - windowStartX)][imageStartY + (y - windowStartY)];
            }  
        }
    }
    return 0;   
}



