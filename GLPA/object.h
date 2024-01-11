#ifndef OBJECT_H_
#define OBJECT_H_

#include <string>
#include <unordered_map>

#include "mesh.h"
#include "error.h"

class Object{
public :
    std::wstring name;
    std::unordered_map<std::string, Mesh> mesh;

    void loadMesh(std::string file_name, std::string folder_pass);
};


#endif OBJECT_H_


