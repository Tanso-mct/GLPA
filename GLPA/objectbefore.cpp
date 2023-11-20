#include "objectbefore.h"

void OBJECT::inputData(std::string inputFileName)
{
    OBJ_FILE objFile;
    objFile.loadData(inputFileName);
    data.push_back(objFile);
}

OBJECT tempObject;
