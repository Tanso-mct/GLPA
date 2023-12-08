#include "png.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool Png::load(std::string filePath){
    int lastSolidus = filePath.rfind("/");
    name = filePath.substr(lastSolidus, filePath.length());

    stbi_uc* pixels = stbi_load(name.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        // �ǂݍ��݃G���[�����������ꍇ�̏���
        OutputDebugStringW(_T("GLPA : ERROR"));
    }

    // 2. �s�N�Z���f�[�^��LPDWORD�^�ϐ��ɕϊ�
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

    // �s�N�Z���f�[�^�̎g�p���I���������
    stbi_image_free(pixels);
    return false;
}
