#include "File.h"

Glpa::File::File()
{
    type = Glpa::CLASS_FILE;
}

Glpa::File::~File()
{
}

void Glpa::File::setFilePath(std::string str)
{
    fileName = str.substr(str.rfind("/") + 1, str.size() - str.rfind("/"));
    filePath = str;
}