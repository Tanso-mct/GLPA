#ifndef MESH_H_
#define MESH_H_


#include <string>
#include <iostream>
#include <fstream>

#include <locale>
#include <codecvt>


#include "cg.h"
#include "error.h"

class Mesh{
public :
    RangeRect range;
    Vertices v;
    Polygons poly;

    std::wstring name;
    std::string filePath;

    void load(std::string file_name, std::string folder_pass);

};


#endif MESH_H_
