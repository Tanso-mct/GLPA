#include "view_volume.cuh"

void ViewVolume::pushFaceVsToDouble(std::vector<double>* targetVec, int faceI){
    if (faceI == SURFACE_TOP){
        vec.pushVecToDouble(v, targetVec, 0);
        vec.pushVecToDouble(v, targetVec, 4);
        vec.pushVecToDouble(v, targetVec, 5);
        vec.pushVecToDouble(v, targetVec, 1);
        vec.pushVecToDouble(v, targetVec, 0);
        vec.pushVecToDouble(v, targetVec, 4);
    }
    else if (faceI == SURFACE_FRONT){
        vec.pushVecToDouble(v, targetVec, 3);
        vec.pushVecToDouble(v, targetVec, 0);
        vec.pushVecToDouble(v, targetVec, 1);
        vec.pushVecToDouble(v, targetVec, 2);
        vec.pushVecToDouble(v, targetVec, 3);
        vec.pushVecToDouble(v, targetVec, 0);
    }
    else if (faceI == SURFACE_RIGHT){
        vec.pushVecToDouble(v, targetVec, 2);
        vec.pushVecToDouble(v, targetVec, 1);
        vec.pushVecToDouble(v, targetVec, 5);
        vec.pushVecToDouble(v, targetVec, 6);
        vec.pushVecToDouble(v, targetVec, 2);
        vec.pushVecToDouble(v, targetVec, 1);
    }
    else if (faceI == SURFACE_LEFT){
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 4);
        vec.pushVecToDouble(v, targetVec, 0);
        vec.pushVecToDouble(v, targetVec, 3);
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 4);
    }
    else if (faceI == SURFACE_BACK){
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 4);
        vec.pushVecToDouble(v, targetVec, 5);
        vec.pushVecToDouble(v, targetVec, 6);
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 4);
    }
    else if (faceI == SURFACE_BOTTOM){
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 3);
        vec.pushVecToDouble(v, targetVec, 2);
        vec.pushVecToDouble(v, targetVec, 6);
        vec.pushVecToDouble(v, targetVec, 7);
        vec.pushVecToDouble(v, targetVec, 3);
    }
    
}