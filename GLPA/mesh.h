#ifndef MESH_H_
#define MESH_H_

#include "file.h"
#include "cgmath.cuh"

#include <string>
#include <vector>
class MESH
{
public :
    // Object File data
    std::vector<OBJ_FILE> data;

    // Class data used for loading
    void inputData(std::string inputFileName);
};

extern MESH tempMesh;

#endif MESH_H_
