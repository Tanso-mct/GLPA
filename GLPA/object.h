#ifndef OBJECT_H_
#define OBJECT_H_

#include "file.h"
#include "cgmath.cuh"

#include <string>
#include <vector>
class OBJECT
{
public :
    // Object File data
    std::vector<OBJ_FILE> data;

    // Class data used for loading
    OBJ_FILE objFile;
    void inputData(std::string inputFileName);
};

#endif OBJECT_H_
