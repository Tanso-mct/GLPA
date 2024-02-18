#ifndef OBJ_H_
#define OBJ_H_


#include <string>
#include <iostream>
#include <fstream>

#include <locale>
#include <codecvt>


#include "cg.h"
#include "error.h"

class Obj{
public :
    RangeRect range;
    Vertices v;
    Polygons poly;

    std::wstring name;
    std::string filePath;

    void load(std::string file_name, std::string folder_pass);

};


#endif OBJ_H_
