#include "object.h"


void Object::loadMesh(std::string fileName, std::string folderPass){
    if (mesh.find(fileName) != mesh.end()){
        std::runtime_error(ERROR_OBJECT_LOAD);
    }

    mesh[fileName].load(fileName, folderPass);

    for (int i = 0; i < 8; i++){
        if (range.status){
            if (mesh[fileName].range.wVertex[i].x < range.origin.x){
                range.origin.x = mesh[fileName].range.wVertex[i].x;
            }
            if (mesh[fileName].range.wVertex[i].y < range.origin.y){
                range.origin.y = mesh[fileName].range.wVertex[i].y;
            }
            if (mesh[fileName].range.wVertex[i].z > range.origin.z){
                range.origin.z = mesh[fileName].range.wVertex[i].z;
            }

            // Processing with respect to opposite point
            if (mesh[fileName].range.wVertex[i].x > range.opposite.x){
                range.opposite.x = mesh[fileName].range.wVertex[i].x;
            }
            if (mesh[fileName].range.wVertex[i].y > range.opposite.y){
                range.opposite.y = mesh[fileName].range.wVertex[i].y;
            }
            if (mesh[fileName].range.wVertex[i].z < range.opposite.z){
                range.opposite.z = mesh[fileName].range.wVertex[i].z;
            }
        }
        else{
            range.origin.x = mesh[fileName].range.wVertex[i].x;
            range.origin.y = mesh[fileName].range.wVertex[i].y;
            range.origin.z = mesh[fileName].range.wVertex[i].z;

            range.opposite.x = mesh[fileName].range.wVertex[i].x;
            range.opposite.y = mesh[fileName].range.wVertex[i].y;
            range.opposite.z = mesh[fileName].range.wVertex[i].z;
            range.status = true;
        }
    }

    range.wVertex.resize(8);

    range.wVertex[0] = {range.origin.x, range.opposite.y, range.origin.z};
    range.wVertex[1] = {range.opposite.x, range.opposite.y, range.origin.z};
    range.wVertex[2] = {range.opposite.x, range.origin.y, range.origin.z};
    range.wVertex[3] = {range.origin.x, range.origin.y, range.origin.z};
    range.wVertex[4] = {range.origin.x, range.opposite.y, range.opposite.z};
    range.wVertex[5] = {range.opposite.x, range.opposite.y, range.opposite.z};
    range.wVertex[6] = {range.opposite.x, range.origin.y, range.opposite.z};
    range.wVertex[7] = {range.origin.x, range.origin.y, range.opposite.z};
}
