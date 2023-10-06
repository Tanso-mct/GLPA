#include "object.h"

void OBJECT::inputData(std::string inputFileName)
{
    objFile.loadData(inputFileName);
    data.push_back(objFile);
}

OBJECT tempObject;
