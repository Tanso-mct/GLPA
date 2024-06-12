#include "Color.h"

void Glpa::Color::SetRbga(DWORD buf)
{
    sourceA = (buf >> 24) & 0xFF;
    sourceR = (buf >> 16) & 0xFF;
    sourceG = (buf >> 8) & 0xFF;
    sourceB = buf & 0xFF;
}

void Glpa::Color::AlphaBlend(Glpa::Color bg)
{
    
}
