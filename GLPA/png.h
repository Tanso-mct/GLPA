#ifndef PNG_H_
#define PNG_H_

#include <FreeImage.h>

// PNG画像を読み込んでLPDWORDに変換する関数
LPDWORD LoadPNGImage(const char* filename, int& width, int& height) {
    FreeImage_Initialise();

    // 画像を読み込む
    FIBITMAP* image = FreeImage_Load(FIF_PNG, filename, PNG_DEFAULT);

    if (!image) {
        // エラー処理
        FreeImage_DeInitialise();
        return nullptr;
    }

    // 画像の幅と高さを取得
    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);

    // ピクセルデータを取得
    BYTE* bits = FreeImage_GetBits(image);

    // LPDWORD型に変換
    size_t imageSize = width * height * 4;  // 4はRGBAの各要素のバイト数
    LPDWORD pixelData = (LPDWORD)malloc(imageSize);
    memcpy(pixelData, bits, imageSize);

    // 画像の解放
    FreeImage_Unload(image);

    return pixelData;
}
class Png
{
public :


private :
};

#endif PNG_H_


