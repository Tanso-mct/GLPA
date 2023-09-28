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
    VECTOR3D* sourceMatrices, 
    double* calcMatrices, 
    double* resultMatrices, 
    int n,
    int matrixRaw
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < matrixRaw)
    {
        resultMatrices[matrixRaw*i+j] 
        = sourceMatrices[i].x * calcMatrices[j+matrixRaw*R1] 
        + sourceMatrices[i].y * calcMatrices[j+matrixRaw*R2]
        + sourceMatrices[i].z * calcMatrices[j+matrixRaw*R3];
    }

}

__global__ void gpuCalc4xMatrixProduct
(
    VECTOR3D *sourceMatrices, 
    double *calcMatrices, 
    double *resultMatrices, 
    int n,
    int matrixRaw
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < matrixRaw)
    {
        resultMatrices[matrixRaw*i+j] 
        = sourceMatrices[i].x * calcMatrices[j+matrixRaw*R1] 
        + sourceMatrices[i].y * calcMatrices[j+matrixRaw*R2]
        + sourceMatrices[i].z * calcMatrices[j+matrixRaw*R3]
        + 1 * calcMatrices[j+matrixRaw*R4];
    }
}

void MATRIX::calcMatrix3xProduct()
{
    // Allocate memory for each matrix size
    hSourceMatrices = (VECTOR3D *)malloc(sizeof(VECTOR3D)*sourceMatrices.size());
    hCalcMatrices = (double *)malloc(sizeof(double)*matrixRaw*calcMatrices3x.size());
    hResultMatrices = (double *)calloc(matrixRaw*sourceMatrices.size(), sizeof(double));

    // Copy member variable
    hSourceMatrices = sourceMatrices.data();

    for (int i = 0; i < calcMatrices3x.size(); ++i)
    {
        hCalcMatrices[i*matrixRaw+R1] = calcMatrices3x[i].x;
        hCalcMatrices[i*matrixRaw+R2] = calcMatrices3x[i].y;
        hCalcMatrices[i*matrixRaw+R3] = calcMatrices3x[i].z;
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size());
    cudaMalloc((void**)&dCalcMatrices, sizeof(double)*matrixRaw*calcMatrices3x.size());
    cudaMalloc((void**)&dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSourceMatrices, hSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcMatrices, hCalcMatrices, sizeof(double)*matrixRaw*calcMatrices3x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dResultMatrices, hResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    if (matrixRaw == MATRIX3RAW)
    {
        gpuCalc3xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size(), matrixRaw);
    }
    else if (matrixRaw == MATRIX4RAW)
    {
        gpuCalc4xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size(), matrixRaw);
    }

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    for (int i = 0; i < sizeof(double)*matrixRaw*sourceMatrices.size(); ++i)
    {
        vecSystem.inputVec3d
        (
            hResultMatrices[i*matrixRaw+C1], 
            hResultMatrices[i*matrixRaw+C2], 
            hResultMatrices[i*matrixRaw+C3], 
            i / matrixRaw,
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
    hSourceMatrices = (VECTOR3D *)malloc(sizeof(VECTOR3D)*sourceMatrices.size());
    hCalcMatrices = (double *)malloc(sizeof(double)*matrixRaw*calcMatrices4x.size());
    hResultMatrices = (double *)calloc(matrixRaw*sourceMatrices.size(), sizeof(double));

    // Copy member variable
    hSourceMatrices = sourceMatrices.data();

    for (int i = 0; i < calcMatrices4x.size(); ++i)
    {
        hCalcMatrices[i*matrixRaw+R1] = calcMatrices4x[i].x;
        hCalcMatrices[i*matrixRaw+R2] = calcMatrices4x[i].y;
        hCalcMatrices[i*matrixRaw+R3] = calcMatrices4x[i].z;
        hCalcMatrices[i*matrixRaw+R4] = calcMatrices4x[i].w;
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size());
    cudaMalloc((void**)&dCalcMatrices, sizeof(double)*matrixRaw*calcMatrices4x.size());
    cudaMalloc((void**)&dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSourceMatrices, hSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcMatrices, hCalcMatrices, sizeof(double)*matrixRaw*calcMatrices4x.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dResultMatrices, hResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    if (matrixRaw == MATRIX3RAW)
    {
        gpuCalc3xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size(), matrixRaw);
    }
    else if (matrixRaw == MATRIX4RAW)
    {
        gpuCalc4xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size(), matrixRaw);
    }

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    for (int i = 0; i < sizeof(double)*matrixRaw*sourceMatrices.size(); ++i)
    {
        vecSystem.inputVec3d
        (
            hResultMatrices[i*matrixRaw+C1], 
            hResultMatrices[i*matrixRaw+C2], 
            hResultMatrices[i*matrixRaw+C3],  
            i / matrixRaw,
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
    matrixRaw = MATRIX4RAW;
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

void MATRIX::rotTrans
(
    std::vector<VECTOR3D> sourceCoordinates, 
    int rotationAxis, 
    double rotationAngle
)
{
    matrixRaw = MATRIX3RAW;
    sourceMatrices = sourceCoordinates;

    switch (rotationAxis)
    {
    case SELECTAXIS_X:
        input3xMatrix
        (
            &calcMatrices3x,
            1,    0,                                     0, 
            0,    cos(rotationAngle * PI / 180),    -1 * sin(rotationAngle * PI / 180),
            0,    sin(rotationAngle * PI / 180),    cos(rotationAngle * PI / 180)
        );
        break;

    case SELECTAXIS_Y:
        input3xMatrix
        (
            &calcMatrices3x,
            cos(rotationAngle * PI / 180),         0,     sin(rotationAngle * PI / 180), 
            0,                                          1,     0,
            -1 * sin(rotationAngle * PI / 180),    0,     cos(rotationAngle * PI / 180)
        );
        break;

    case SELECTAXIS_Z:
        input3xMatrix
        (
            &calcMatrices3x,
            cos(rotationAngle * PI / 180),     -1 * sin(rotationAngle * PI / 180),   0, 
            sin(rotationAngle * PI / 180),     cos(rotationAngle * PI / 180),        0,
            0,                                      0,                                         1
        );
        break;
    }
    
    calcMatrix3xProduct();
}

void MATRIX::scaleTrans(std::vector<VECTOR3D> sourceCoordinates, VECTOR3D scalingRate)
{
    matrixRaw = MATRIX3RAW;
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
