#ifndef MESH_H_
#define MESH_H_


#include <string>
#include <iostream>
#include <fstream>


#include "cg.h"
#include "error.h"

class Mesh{
public :
    RangeRect range;
    Vertex v;
    Polygons poly;

    void load(std::string file_name, std::string folder_pass);

};


#endif MESH_H_
