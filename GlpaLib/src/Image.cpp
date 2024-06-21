#include "Image.h"

Glpa::Image::Image(std::string argName, std::string filePath, Glpa::Vec2d defPos) : pos(defPos)
{
    name = argName;
    setFilePath(filePath);

    type = Glpa::CLASS_IMAGE;
}

Glpa::Image::~Image()
{
}