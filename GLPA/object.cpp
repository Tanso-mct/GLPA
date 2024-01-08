#include "object.h"


void Object::loadMesh(std::string fileName, std::string folderPass){
    Mesh tempMesh;
    tempMesh.load(fileName, folderPass);

    mesh[fileName] = tempMesh;
}
