#ifndef CGMATH_H_
#define CGMATH_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <vector>
#include <algorithm>
#include <Windows.h>

// Math
#define PI 3.14159265

// Column number
#define C1 0
#define C2 1
#define C3 2
#define C4 3

// Row number
#define R1 0
#define R2 1
#define R3 2
#define R4 3

// Matrix raw amout
#define MATRIX3RAW 3
#define MATRIX4RAW 4

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

typedef struct tagVECTOR4D
{
    double x;
    double y;
    double z;
    double w;
} VECTOR4D;

typedef struct tagVECTOR_XZ
{
    double x;
    double z;
} VECTOR_XZ;

typedef struct tagVECTOR_YZ
{
    double y;
    double z;
} VECTOR_YZ;

typedef struct tagSIZE2
{
    double width;
    double height;
} SIZE2;

typedef struct tagRANGE_CUBE
{
    bool status = false;
    VECTOR3D origin;
    VECTOR3D opposite;
    std::vector<VECTOR3D> wVertex;
} RANGE_CUBE;

__global__ void gpuVecDotProduct
(
    double* source_vector, 
    double* calc_vector, 
    double* result_vector, 
    int size_n // Number of array columns
);

class VECTOR
{
public :
    // data
    std::vector<VECTOR3D> souceVector;
    VECTOR3D calcVector;
    std::vector<VECTOR3D> resultVector;

    // host memory
    double* hSouceVec;
    double* hCalcVec;
    double* hResultVec;

    // device memory
    double* dSouceVec;
    double* dCalcVec;
    double* dResultVec;

    void pushVec3d
    (
        double push_x,
        double push_y,
        double push_z,
        std::vector<VECTOR3D>* input_vevotr3d
    );

    void pushVec4d
    (
        double push_x,
        double push_y,
        double push_z,
        double push_w,
        std::vector<VECTOR4D>* input_vevotr3d
    );

    void inputVec3d
    (
        double input_x,
        double input_y,
        double input_z,
        int array_num_input,
        std::vector<VECTOR3D>* input_vevotr3d
    );

    void inputVec4d
    (
        double input_x,
        double input_y,
        double input_z,
        double input_w,
        int arrayNumInput,
        std::vector<VECTOR4D>* input_vevotr4d
    );

    void dotProduct
    (
        std::vector<VECTOR3D> source_vector,
        std::vector<VECTOR3D> calc_vector
    );
};


// 3x3 matrix product
__global__ void gpuCalc3xMatrixProduct
(
    double* source_matrices, 
    double* calc_matrices, 
    double* result_matrices, 
    int size_n // Number of array columns
);

__global__ void gpuCalc4xMatrixProduct
(
    double* source_matrices, 
    double* calc_matrices, 
    double* result_matrices, 
    int size_n // Number of array columns
);

class MATRIX
{
public :
    // vector class
    VECTOR vec;
    
    // vector data
    std::vector<VECTOR3D> sourceMatrices; 
    std::vector<VECTOR3D> calcMatrices3x;
    std::vector<VECTOR4D> calcMatrices4x;
    std::vector<VECTOR3D> resultMatrices;

    MATRIX()
    {
        // std::vectorの配列サイズの指定
        sourceMatrices.resize(3);
        calcMatrices3x.resize(3);
        calcMatrices4x.resize(4);
        resultMatrices.resize(3);
    }

    // host memory
    double* hSourceMatrices; // Value of 3D coordinates before calculation
    double* hCalcMatrices;   // Matrix value to be used in the calculation
    double* hResultMatrices; // Calculated 3D Coordinate 

    // device memory
    double* dSourceMatrices;
    double* dCalcMatrices;
    double* dResultMatrices;

    void input3xMatrix
    (
        std::vector<VECTOR3D>* inputMatrix,
        double a_11, double a_12, double a_13,
        double a_21, double a_22, double a_23,
        double a_31, double a_32, double a_33
    );

    void input4xMatrix
    (
        std::vector<VECTOR4D>* inputMatrix,
        double a_11, double a_12, double a_13, double a_14,
        double a_21, double a_22, double a_23, double a_24,
        double a_31, double a_32, double a_33, double a_34,
        double a_41, double a_42, double a_43, double a_44
    );

    // Enter and use numerical values for each of sourceMatrix, calcMatrix, and matrixRaw.
    void calcMatrix3xProduct();
    void calcMatrix4xProduct();

    // Calculation of each matrix
    void posTrans
    (
        std::vector<VECTOR3D> source_coordinates,
        VECTOR3D change_pos_amount
    );

    void rotTrans
    (
        std::vector<VECTOR3D> source_3d_coordinates,
        VECTOR3D rotation_angle
    );

    void scaleTrans
    (
        std::vector<VECTOR3D> source_3d_coordinates,
        VECTOR3D scaling_rate
    );

};

#endif CGMATH_H_
