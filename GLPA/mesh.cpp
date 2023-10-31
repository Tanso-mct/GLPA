#include "mesh.h"

void MESH::inputData(std::string inputFileName)
{
    OBJ_FILE objFile;
    objFile.loadData(inputFileName);
    data.push_back(objFile);
}

MESH tempMesh;
