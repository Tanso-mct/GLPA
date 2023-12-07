#ifndef PNG_H_
#define PNG_H_

#include <FreeImage.h>

// PNG�摜��ǂݍ����LPDWORD�ɕϊ�����֐�
LPDWORD LoadPNGImage(const char* filename, int& width, int& height) {
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
class Png
{
public :


private :
};

#endif PNG_H_


