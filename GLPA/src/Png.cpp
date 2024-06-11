#include "Png.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Glpa::Png::Png()
{
    type = Glpa::CLASS_PNG;
}

Glpa::Png::~Png()
{
}

void Glpa::Png::load()
{
    stbi_uc* pixels = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        OutputDebugStringA("GlpaLib ERROR Png.cpp - Failed to load PNG image.\n");
        throw std::runtime_error("Failed to load PNG image.");
        return;
    }

    data = new DWORD[width * height];

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

    stbi_image_free(pixels);
    loaded = true;
}

void Glpa::Png::release()
{
    delete data;
    loaded = false;
}
