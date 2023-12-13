#include "buffer_2d.cuh"

// // RGBAデータを格納する構造体
// struct Pixel {
//     unsigned char r, g, b, a; // Red, Green, Blue, Alpha
// };

// // アルファブレンディング関数
// void AlphaBlend(const Pixel& srcPixel, const Pixel& destPixel, Pixel& resultPixel) {
//     float alpha = static_cast<float>(srcPixel.a) / 255.0f;
//     float invAlpha = 1.0f - alpha;

//     resultPixel.r = static_cast<unsigned char>(alpha * srcPixel.r + invAlpha * destPixel.r);
//     resultPixel.g = static_cast<unsigned char>(alpha * srcPixel.g + invAlpha * destPixel.g);
//     resultPixel.b = static_cast<unsigned char>(alpha * srcPixel.b + invAlpha * destPixel.b);
//     resultPixel.a = static_cast<unsigned char>(srcPixel.a + invAlpha * destPixel.a);
// }

// // 上に重ねる画像と下にある画像のピクセルを取得
// Pixel srcPixel, destPixel, resultPixel;

// srcPixel.r = (srcImageData[index] >> 16) & 0xFF;
// srcPixel.g = (srcImageData[index] >> 8) & 0xFF;
// srcPixel.b = srcImageData[index] & 0xFF;
// srcPixel.a = (srcImageData[index] >> 24) & 0xFF;

// destPixel.r = (destImageData[index] >> 16) & 0xFF;
// destPixel.g = (destImageData[index] >> 8) & 0xFF;
// destPixel.b = destImageData[index] & 0xFF;
// destPixel.a = (destImageData[index] >> 24) & 0xFF;

// // アルファブレンディングを適用
// AlphaBlend(srcPixel, destPixel, resultPixel);

// // 最終的なRGBA値をLPDWORDに格納
// LPDWORD finalPixel = &srcImageData[index]; // 上に重ねる画像のデータを更新する場合は変更が必要

// *finalPixel = (resultPixel.a << 24) | (resultPixel.r << 16) | (resultPixel.g << 8) | resultPixel.b;