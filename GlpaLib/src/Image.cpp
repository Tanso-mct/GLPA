#include "Image.h"

Glpa::Image::Image(std::string argName, std::string argFilePath, Glpa::Vec2d defPos) : pos(defPos)
{
    name = argName;
    type = Glpa::CLASS_IMAGE;

    filePath = argFilePath;
}

Glpa::Image::~Image()
{

}

void Glpa::Image::load()
{
    fileDataManager->newFile(filePath);
    fileDataManager->load(filePath);
}

void Glpa::Image::release()
{
    fileDataManager->release(filePath);
}
