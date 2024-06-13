#include "Color.h"

Glpa::Color::Color(BYTE r, BYTE g, BYTE b, BYTE a)
{
    sourceR = r;
    sourceG = g;
    sourceB = b;
    sourceA = a;

    nowR = r;
    nowG = g;
    nowB = b;
    nowA = a;
}

Glpa::Color::Color(DWORD buf)
{
    SetRgba(buf);
}

void Glpa::Color::SetRgba(DWORD buf)
{
    sourceA = (buf >> 24) & 0xFF;
    sourceR = (buf >> 16) & 0xFF;
    sourceG = (buf >> 8) & 0xFF;
    sourceB = buf & 0xFF;
}

DWORD Glpa::Color::GetDword()
{
    DWORD rtDword = (nowA << 24) | (nowR << 16) | (nowG << 8) | nowB;
    return rtDword;
}

void Glpa::Color::AlphaBlend(Glpa::Color bg)
{
    float alpha = static_cast<float>(nowA) / 255.0f;
    float invAlpha = 1.0f - alpha;

    nowR = static_cast<unsigned char>(alpha * nowR + invAlpha * bg.GetNowR());
    nowG = static_cast<unsigned char>(alpha * nowG + invAlpha * bg.GetNowG());
    nowB = static_cast<unsigned char>(alpha * nowB + invAlpha * bg.GetNowB());
    nowA = static_cast<unsigned char>(nowA + invAlpha * bg.GetNowA());
}
