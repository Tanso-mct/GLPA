#ifndef CGMATH_H_
#define CGMATH_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <vector>
#include <algorithm>
#include <Windows.h>

//******************************************************************
// Kernel function call-related processing method in 3 dimensions
// int blockSize = 32;
// dim3 dimBlock(blockSize, blockSize, blockSize);
// dim3 dimGrid
// (
//     (xElementsNum + blockSize - 1) / blockSize, 
//     (yElementsNum + blockSize - 1) / blockSize, 
//     (zElementsNum + blockSize - 1) / blockSize
// );

// Two-dimensional kernel function call-related processing method
// dim3 dimBlock(32, 32); // Thread block size
// dim3 dimGrid((xElementsNum + dimBlock.x - 1) / dimBlock.x, 
// (yElementsNum + dimBlock.y - 1) / dimBlock.y); // Grid Size

// One-dimensional kernel function call-related processing method
// int blockSize = 1024;
// int numBlocks = (xElementsNum + blockSize - 1) / blockSize;
// dim3 dimBlock(blockSize, 1, 1);
// dim3 dimGrid(numBlocks, 1, 1);
//******************************************************************

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

// Vector axis
#define VX 0
#define VY 1
#define VZ 2

// Vector 3d
#define VECTOR3 3

// Matrix raw amout
#define MATRIX3RAW 3

// Equiation call name
#define X1 0
#define Y1 1
#define Z1 2

#define LX 0
#define MY 1
#define NZ 2

#define X0 0
#define Y0 1
#define Z0 2

#define PX 0
#define QY 1
#define RZ 2

// Used to see if an intersection exists
#define I_FALSE 0
#define I_TRUE 1

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

typedef struct tagINT2D
{
    std::vector<int> n;
} INT2D;

__global__ void gpuVecAddition
(
    double* source_vector, 
    double* calc_vector, 
    double* result_vector, 
    int size_n // Number of array columns
);

__global__ void gpuVecDotProduct
(
    double* source_vector, 
    double* calc_vector, 
    double* result_vector, 
    int size_n // Number of array columns
);

__global__ void gpuVecCrossProduct
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
    std::vector<double> resultVector;
    std::vector<VECTOR3D> resultVector3D;

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

    void inputVec3d
    (
        double input_x,
        double input_y,
        double input_z,
        int array_num_input,
        std::vector<VECTOR3D>* input_vevotr3d
    );

    void posTrans
    (
        std::vector<VECTOR3D> source_vector,
        VECTOR3D calc_vector
    );

    void dotProduct
    (
        std::vector<VECTOR3D> source_vector,
        std::vector<VECTOR3D> calc_vector
    );

    void crossProduct
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

class MATRIX
{
public :
    // vector class
    VECTOR vec;
    
    // vector data
    std::vector<VECTOR3D> sourceMatrices; 
    std::vector<VECTOR3D> calcMatrices3x;
    std::vector<VECTOR3D> resultMatrices;

    MATRIX()
    {
        // std::vectorの配列サイズの指定
        sourceMatrices.resize(3);
        calcMatrices3x.resize(3);
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

    // Enter and use numerical values for each of sourceMatrix, calcMatrix, and matrixRaw.
    void calcMatrix3xProduct();

    // Calculation of each matrix
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

__global__ void gpuGetLinePlaneI
(
    double* line_vertex_A,
    double* line_vertex_B,
    double* parametor_t,
    double* line_plane_I,
    int i_amout
);

__global__ void gpuGetDenominateT
(
    double* line_vertex_B,
    double* plane_normal,
    double* parameter_t,
    int intersection_amout,
    int planeAmout
);

class EQUATION
{
public :
    VECTOR vec;
    std::vector<double> paraT;
    std::vector<INT2D> amoutIeachLine;
    std::vector<VECTOR3D> linePlaneI;

    double* hLineVertexA;
    double* hLineVertexB;
    double* hPlaneVertex;
    double* hPlaneNormal;
    double* hParaT;
    double* hLinePlaneI;

    double* dLineVertexA;
    double* dLineVertexB;
    double* dPlaneVertex;
    double* dPlaneNormal;
    double* dParaT;
    double* dLinePlaneI;

    void getDenominateT
    (
        std::vector<VECTOR3D> line_vertex_B, // l, m, n
        std::vector<VECTOR3D> plane_normal // p, q, r
    );

    void getLinePlaneI
    (
        std::vector<VECTOR3D> line_vertex_A, // x1, y1, z1
        std::vector<VECTOR3D> line_vertex_B, // l, m, n
        std::vector<VECTOR3D> plane_vertex, // x0, y0, z0
        std::vector<VECTOR3D> plane_normal // p, q, r
    );

};

#endif CGMATH_H_
