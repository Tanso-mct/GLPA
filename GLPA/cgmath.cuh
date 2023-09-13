#ifndef CGMATH_H_
#define CGMATH_H_

#include <vector>
#include <algorithm>

// Column number
#define C1 0
#define C2 1
#define C3 2

// Row number
#define R1 0
#define R2 1
#define R3 2

// Selected axis
#define SELECTAXIS_X 0
#define SELECTAXIS_Y 1
#define SELECTAXIS_Z 2

// Matrix raw amout
#define MATRIX3RAW 3
#define MATRIX4RAW 4

// Number of threads per block
#define BS 1024

typedef struct tagVECTOR2D
{
    double x;
    double y;
} VECTOR2D;

typedef struct tagVECTOR3D
{
    double x;
    double y;
    double z;
} VECTOR3D;

class MATRIX
{
public :
    // vector data
    std::vector<VECTOR3D> sourceMatrices;
    std::vector<VECTOR3D> calcMatrices;
    std::vector<VECTOR3D> resultMatrices;

    // host memory
    VECTOR3D* hSourceMatrices; // Value of 3D coordinates before calculation
    double* hCalcMatrices;   // Matrix value to be used in the calculation
    double* hResultMatrices; // Calculated 3D Coordinate 

    // device memory
    VECTOR3D* dSourceMatrices;
    double* dCalcMatrices;
    double* dResultMatrices;

    int matrixRaw;

    // 3x3 matrix product
    __global__ void gpuCalcMatrixProduct
    (
        VECTOR3D* source_matrices, 
        double* calc_matrices, 
        double* result_matrices, 
        int n // Number of array columns
    );

    void calcMatrixProduct();

    // Calculation of each matrix
    void posTrans
    (
        std::vector<VECTOR3D> source_coordinates,
        VECTOR3D change_pos_amount
    );

    void rotTrans
    (
        std::vector<VECTOR3D> source_3d_coordinates,
        int rotation_axis,
        double rotation_angle
    );

    void scaleTrans
    (
        std::vector<VECTOR3D> source_3d_coordinates,
        VECTOR3D scaling_rate
    );

};

#endif CGMATH_H_
