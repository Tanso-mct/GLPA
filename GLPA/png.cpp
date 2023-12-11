#include "png.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool Png::load(std::string filePath){
    std::size_t lastSolidus = filePath.rfind("/");

    if (lastSolidus != std::string::npos){
        name = filePath.substr(lastSolidus+1, filePath.find(".") - (lastSolidus+1));
    }
    else
    {
        name = filePath;
    }

    stbi_uc* pixels = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        throw std::runtime_error(ERROR_PNG_LOAD);
        return false;
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
    return true;
}

void Png::release(){
    delete(data);
}
