#include "File.h"

void Glpa::File::setFilePath(std::string str)
{
    fileName = str.substr(str.rfind("/") + 1, str.size() - str.rfind("/"));
    filePath = str;
}