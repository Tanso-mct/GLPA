#include "buffer_2d.cuh"

// // RGBA�f�[�^���i�[����\����
// struct Pixel {
//     unsigned char r, g, b, a; // Red, Green, Blue, Alpha
// };

// // �A���t�@�u�����f�B���O�֐�
// void AlphaBlend(const Pixel& srcPixel, const Pixel& destPixel, Pixel& resultPixel) {
//     float alpha = static_cast<float>(srcPixel.a) / 255.0f;
//     float invAlpha = 1.0f - alpha;

//     resultPixel.r = static_cast<unsigned char>(alpha * srcPixel.r + invAlpha * destPixel.r);
//     resultPixel.g = static_cast<unsigned char>(alpha * srcPixel.g + invAlpha * destPixel.g);
//     resultPixel.b = static_cast<unsigned char>(alpha * srcPixel.b + invAlpha * destPixel.b);
//     resultPixel.a = static_cast<unsigned char>(srcPixel.a + invAlpha * destPixel.a);
// }

// // ��ɏd�˂�摜�Ɖ��ɂ���摜�̃s�N�Z�����擾
// Pixel srcPixel, destPixel, resultPixel;

// srcPixel.r = (srcImageData[index] >> 16) & 0xFF;
// srcPixel.g = (srcImageData[index] >> 8) & 0xFF;
// srcPixel.b = srcImageData[index] & 0xFF;
// srcPixel.a = (srcImageData[index] >> 24) & 0xFF;

// destPixel.r = (destImageData[index] >> 16) & 0xFF;
// destPixel.g = (destImageData[index] >> 8) & 0xFF;
// destPixel.b = destImageData[index] & 0xFF;
// destPixel.a = (destImageData[index] >> 24) & 0xFF;

// // �A���t�@�u�����f�B���O��K�p
// AlphaBlend(srcPixel, destPixel, resultPixel);

// // �ŏI�I��RGBA�l��LPDWORD�Ɋi�[
// LPDWORD finalPixel = &srcImageData[index]; // ��ɏd�˂�摜�̃f�[�^���X�V����ꍇ�͕ύX���K�v

// *finalPixel = (resultPixel.a << 24) | (resultPixel.r << 16) | (resultPixel.g << 8) | resultPixel.b;