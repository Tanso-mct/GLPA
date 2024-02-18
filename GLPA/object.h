#ifndef OBJECT_H_
#define OBJECT_H_

#include <string>
#include <unordered_map>

#include <iostream>
#include <fstream>

#include <locale>
#include <codecvt>

#include "cg.h"
#include "error.h"

class Object{
public :
    RangeRect range;
    Vertices v;
    Polygons poly;

    std::wstring name;
    std::string filePath;

    void load(std::wstring file_name, std::string folder_pass);
};


#endif OBJECT_H_


