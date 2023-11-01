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
    void inputData(std::string inputFileName);
};

extern OBJECT tempObject;

#endif OBJECT_H_
