#include "object.h"


void Object::loadMesh(std::string fileName, std::string folderPass){
    if (mesh.find(fileName) != mesh.end()){
        std::runtime_error(ERROR_OBJECT_LOAD);
    }

    mesh[fileName].load(fileName, folderPass);
}
