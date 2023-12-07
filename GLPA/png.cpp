#include "png.h"

#include <FreeImage.h>

LPDWORD Png::load(const char *filename){
    FreeImage_Initialise();

    // �摜��ǂݍ���
    FIBITMAP* image = FreeImage_Load(FIF_PNG, filename, PNG_DEFAULT);

    if (!image) {
        // �G���[����
        FreeImage_DeInitialise();
        return nullptr;
    }

    // �摜�̕��ƍ������擾
    width = FreeImage_GetWidth(image);
    height = FreeImage_GetHeight(image);

    // �s�N�Z���f�[�^���擾
    BYTE* bits = FreeImage_GetBits(image);

    // LPDWORD�^�ɕϊ�
    size_t imageSize = width * height * 4;  // 4��RGBA�̊e�v�f�̃o�C�g��
    LPDWORD pixelData = (LPDWORD)malloc(imageSize);
    memcpy(pixelData, bits, imageSize);

    // �摜�̉��
    FreeImage_Unload(image);

    return pixelData;
}