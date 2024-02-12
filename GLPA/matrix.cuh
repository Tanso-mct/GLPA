#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>

#include "cg.h"


class Matrix{
public :
    // host memory
    double* hSourceMatrices;
    double* hCalcMatrices;
    double* hResultMatrices;

    // device memory
    double* dSourceMatrices;
    double* dCalcMatrices;
    double* dResultMatrices;

    std::vector<Vec3d> transRotConvert(Vec3d trans_vec, Vec3d rot_angle, std::vector<Vec3d> source_vec);
};


#endif MATRIX_H_

