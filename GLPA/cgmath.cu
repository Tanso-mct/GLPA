#include "cgmath.cuh"

void VECTOR::pushVec3d
(
    double pushX,
    double pushY,
    double pushZ,
    std::vector<VECTOR3D>* inputVevotr3d
)
{
    VECTOR3D pushVec{pushX, pushY, pushZ};
    inputVevotr3d->push_back(pushVec);
}

void VECTOR::pushVec4d
(
    double pushX, 
    double pushY, 
    double pushZ, 
    double pushW, 
    std::vector<VECTOR4D> *inputVevotr4d
)
{
    VECTOR4D pushVec{pushX, pushY, pushZ, pushW};
    inputVevotr4d->push_back(pushVec);
}

void VECTOR::inputVec3d
(
    double inputX, 
    double inputY, 
    double inputZ, 
    int arrayNumInput, 
    std::vector<VECTOR3D>* inputVevotr3d
)
{
    (*inputVevotr3d)[arrayNumInput].x = inputX;
    (*inputVevotr3d)[arrayNumInput].y = inputY;
    (*inputVevotr3d)[arrayNumInput].z = inputZ;
}

void VECTOR::inputVec4d
(
    double inputX, 
    double inputY, 
    double inputZ, 
    double inputW,
    int arrayNumInput, 
    std::vector<VECTOR4D>* inputVevotr4d
)
{
    (*inputVevotr4d)[arrayNumInput].x = inputX;
    (*inputVevotr4d)[arrayNumInput].y = inputY;
    (*inputVevotr4d)[arrayNumInput].z = inputZ;
    (*inputVevotr4d)[arrayNumInput].w = inputW;
}


void MATRIX::input3xMatrix
(
    std::vector<VECTOR3D> *inputMatrix, 
    double a11, double a12, double a13, 
    double a21, double a22, double a23, 
    double a31, double a32, double a33
)
{
    (*inputMatrix)[C1].x = a11;
    (*inputMatrix)[C1].y = a21;
    (*inputMatrix)[C1].z = a31;

    (*inputMatrix)[C2].x = a12;
    (*inputMatrix)[C2].y = a22;
    (*inputMatrix)[C2].z = a32;

    (*inputMatrix)[C3].x = a13;
    (*inputMatrix)[C3].y = a23;
    (*inputMatrix)[C3].z = a33;
}

void MATRIX::input4xMatrix
(
    std::vector<VECTOR4D>* inputMatrix,
    double a11, double a12, double a13, double a14,
    double a21, double a22, double a23, double a24,
    double a31, double a32, double a33, double a34,
    double a41, double a42, double a43, double a44
)
{
    (*inputMatrix)[C1].x = a11;
    (*inputMatrix)[C1].y = a21;
    (*inputMatrix)[C1].z = a31;
    (*inputMatrix)[C1].w = a41;

    (*inputMatrix)[C2].x = a12;
    (*inputMatrix)[C2].y = a22;
    (*inputMatrix)[C2].z = a32;
    (*inputMatrix)[C2].w = a42;

    (*inputMatrix)[C3].x = a13;
    (*inputMatrix)[C3].y = a23;
    (*inputMatrix)[C3].z = a33;
    (*inputMatrix)[C3].w = a43;

    (*inputMatrix)[C4].x = a14;
    (*inputMatrix)[C4].y = a24;
    (*inputMatrix)[C4].z = a34;
    (*inputMatrix)[C4].w = a44;    
}



__global__ void gpuCalc3xMatrixProduct
(
    double* sourceMatrices, 
    double* calcMatrices, 
    double* resultMatrices, 
    int size
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && j < MATRIX3RAW)
    {
        resultMatrices[MATRIX3RAW*i + j] 
        = sourceMatrices[i*MATRIX3RAW + C1] * calcMatrices[j + MATRIX3RAW*R1] 
        + sourceMatrices[i*MATRIX3RAW + C2] * calcMatrices[j + MATRIX3RAW*R2]
        + sourceMatrices[i*MATRIX3RAW + C3] * calcMatrices[j + MATRIX3RAW*R3];
    }

}

__global__ void gpuCalc4xMatrixProduct
(
    double *sourceMatrices, 
    double *calcMatrices, 
    double *resultMatrices, 
    int size
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size  && j < MATRIX4RAW)
    {
        resultMatrices[MATRIX3RAW*i + j] 
        = sourceMatrices[i*MATRIX3RAW + C1] * calcMatrices[j + MATRIX4RAW*R1] 
        + sourceMatrices[i*MATRIX3RAW + C2] * calcMatrices[j + MATRIX4RAW*R2]
        + sourceMatrices[i*MATRIX3RAW + C3] * calcMatrices[j + MATRIX4RAW*R3]
        + 1 * calcMatrices[j + MATRIX4RAW*R4];
    }
}

void MATRIX::calcMatrix3xProduct()
{
    // Allocate memory for each matrix size
    hSourceMatrices = (double*)malloc(sizeof(double)*MATRIX3RAW*sourceMatrices.size());
    hCalcMatrices = (double*)malloc(sizeof(double)*MATRIX3RAW*calcMatrices3x.size());
    hResultMatrices = (double*)malloc(sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    // Copy member variable
    memcpy(hSourceMatrices, sourceMatrices.data(), sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    for (int i = 0; i < calcMatrices3x.size(); ++i)
    {
        hCalcMatrices[i*MATRIX3RAW+C1] = calcMatrices3x[i].x;
        hCalcMatrices[i*MATRIX3RAW+C2] = calcMatrices3x[i].y;
        hCalcMatrices[i*MATRIX3RAW+C3] = calcMatrices3x[i].z;
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSourceMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size());
    cudaMalloc((void**)&dCalcMatrices, sizeof(double)*MATRIX3RAW*calcMatrices3x.size());
    cudaMalloc((void**)&dResultMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSourceMatrices, hSourceMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcMatrices, hCalcMatrices, sizeof(double)*MATRIX3RAW*calcMatrices3x.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(dResultMatrices, hResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 blockSize(32, 32); // Thread block size
    dim3 gridSize((MATRIX3RAW + blockSize.x - 1) / blockSize.x, (MATRIX3RAW + blockSize.y - 1) / blockSize.y); // Grid Size
    gpuCalc3xMatrixProduct<<<gridSize, blockSize>>>
    (dSourceMatrices, dCalcMatrices, dResultMatrices, sourceMatrices.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    resultMatrices.resize(sourceMatrices.size());
    for (int i = 0; i < sourceMatrices.size(); ++i)
    {
        vecSystem.inputVec3d
        (
            hResultMatrices[i*MATRIX3RAW+C1], 
            hResultMatrices[i*MATRIX3RAW+C2], 
            hResultMatrices[i*MATRIX3RAW+C3], 
            i,
            &resultMatrices
        );
    }

    // Release all memory allocated by malloc
    free(hSourceMatrices);
    free(hCalcMatrices);
    free(hResultMatrices);

    cudaFree(dSourceMatrices);
    cudaFree(dCalcMatrices);
    cudaFree(dResultMatrices);
}

void MATRIX::calcMatrix4xProduct()
{
    // Allocate memory for each matrix size
    hSourceMatrices = (double*)malloc(sizeof(double)*MATRIX3RAW*sourceMatrices.size());
    hCalcMatrices = (double*)malloc(sizeof(double)*MATRIX4RAW*calcMatrices4x.size());
    hResultMatrices = (double*)malloc(sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    // Copy member variable
    memcpy(hSourceMatrices, sourceMatrices.data(), sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    for (int i = 0; i < calcMatrices4x.size(); ++i)
    {
        hCalcMatrices[i*MATRIX4RAW+R1] = calcMatrices4x[i].x;
        hCalcMatrices[i*MATRIX4RAW+R2] = calcMatrices4x[i].y;
        hCalcMatrices[i*MATRIX4RAW+R3] = calcMatrices4x[i].z;
        hCalcMatrices[i*MATRIX4RAW+R4] = calcMatrices4x[i].w;
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSourceMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size());
    cudaMalloc((void**)&dCalcMatrices, sizeof(double)*MATRIX4RAW*calcMatrices4x.size());
    cudaMalloc((void**)&dResultMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSourceMatrices, hSourceMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcMatrices, hCalcMatrices, sizeof(double)*MATRIX4RAW*calcMatrices4x.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 blockSize(32, 32); // Thread block size
    dim3 gridSize((MATRIX4RAW + blockSize.x - 1) / blockSize.x, (MATRIX4RAW + blockSize.y - 1) / blockSize.y); // Grid Size
    gpuCalc4xMatrixProduct<<<gridSize, blockSize>>>
    (dSourceMatrices, dCalcMatrices, dResultMatrices, sourceMatrices.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    resultMatrices.resize(sourceMatrices.size());
    for (int i = 0; i < sourceMatrices.size(); ++i)
    {
        vecSystem.inputVec3d
        (
            hResultMatrices[i*MATRIX3RAW+C1], 
            hResultMatrices[i*MATRIX3RAW+C2], 
            hResultMatrices[i*MATRIX3RAW+C3],  
            i,
            &resultMatrices
        );
    }

    // Release all memory allocated by malloc
    free(hSourceMatrices);
    free(hCalcMatrices);
    free(hResultMatrices);

    cudaFree(dSourceMatrices);
    cudaFree(dCalcMatrices);
    cudaFree(dResultMatrices);
}

void MATRIX::posTrans(std::vector<VECTOR3D> sourceCoordinates, VECTOR3D changePosAmount)
{
    sourceMatrices.resize(sourceCoordinates.size());
    sourceMatrices = sourceCoordinates;

    input4xMatrix
    (
        &calcMatrices4x,
        1, 0, 0, changePosAmount.x,
        0, 1, 0, changePosAmount.y,
        0, 0, 1, changePosAmount.z,
        0, 0, 0, 0
    );

    calcMatrix4xProduct();
}

void MATRIX::rotTrans(std::vector<VECTOR3D> sourceCoordinates, VECTOR3D rotAngle)
{
    sourceMatrices.resize(sourceCoordinates.size());
    sourceMatrices = sourceCoordinates;

    double calcRotAngle;

    calcRotAngle = rotAngle.x;
    input3xMatrix
    (
        &calcMatrices3x,
        1,    0,                               0, 
        0,    cos(calcRotAngle * PI / 180),    -sin(calcRotAngle * PI / 180),
        0,    sin(calcRotAngle * PI / 180),    cos(calcRotAngle * PI / 180)
    );
    calcMatrix3xProduct();
    sourceMatrices = resultMatrices;

    calcRotAngle = rotAngle.y;
    input3xMatrix
    (
        &calcMatrices3x,
        cos(calcRotAngle * PI / 180),     0,     sin(calcRotAngle * PI / 180), 
        0,                                1,    0,
        -sin(calcRotAngle * PI / 180),    0,     cos(calcRotAngle * PI / 180)
    );
    calcMatrix3xProduct();
    sourceMatrices = resultMatrices;
        
    calcRotAngle = rotAngle.z;
    input3xMatrix
    (
        &calcMatrices3x,
        cos(calcRotAngle * PI / 180),     -sin(calcRotAngle * PI / 180),   0, 
        sin(calcRotAngle * PI / 180),     cos(calcRotAngle * PI / 180),    0,
        0,                                0,                               1
    );
    calcMatrix3xProduct();
    sourceMatrices = resultMatrices;

}

void MATRIX::scaleTrans(std::vector<VECTOR3D> sourceCoordinates, VECTOR3D scalingRate)
{
    sourceMatrices.resize(sourceCoordinates.size());
    sourceMatrices = sourceCoordinates;

    input3xMatrix
    (
        &calcMatrices3x,
        scalingRate.x,  0,              0,
        0,              scalingRate.y,  0,
        0,              0,              scalingRate.z
    );

    calcMatrix3xProduct();
}
