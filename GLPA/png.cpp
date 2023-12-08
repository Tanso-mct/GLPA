#include "png.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool Png::load(std::string filePath){
    int lastSolidus = filePath.rfind("/");
    name = filePath.substr(lastSolidus, filePath.length());

    stbi_uc* pixels = stbi_load(name.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        // 読み込みエラーが発生した場合の処理
        OutputDebugStringW(_T("GLPA : ERROR"));
    }

    // 2. ピクセルデータをLPDWORD型変数に変換
    size_t pixelCount = width * height;

    int pixelIndex = 0;

    for(UINT y = 0; y <= height; y++)
    {
        for(UINT x = 0; x <= width; x++)
        {
            if (x < width && y < height)
            {
                data[x+y*width] = (pixels[pixelIndex * 4 + 3] << 24) | 
                                  (pixels[pixelIndex * 4] << 16) | 
                                  (pixels[pixelIndex * 4 + 1] << 8) | 
                                  pixels[pixelIndex * 4 + 2];
                pixelIndex += 1;
            }  
        }
    }

    // ピクセルデータの使用が終わったら解放
    stbi_image_free(pixels);
    return false;
}
