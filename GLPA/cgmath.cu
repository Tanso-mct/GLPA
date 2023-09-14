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


__global__ void MATRIX::gpuCalc3xMatrixProduct
(
    VECTOR3D* sourceMatrices, 
    double* calcMatrices, 
    double* resultMatrices, 
    int n
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

__global__ void MATRIX::gpuCalc4xMatrixProduct
(
    VECTOR3D *sourceMatrices, 
    double *calcMatrices, 
    double *resultMatrices, 
    int n
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
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size());
    }
    else if (matrixRaw == MATRIX4RAW)
    {
        gpuCalc4xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size());
    }

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable

    for (int i = 0; i < sizeof(double)*matrixRaw*sourceMatrices.size(); ++i)
    {
        vecSystem.pushVec3d(hResultMatrices[i+C1], hResultMatrices[i+C2], hResultMatrices[i+C3], &resultMatrices);
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
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size());
    }
    else if (matrixRaw == MATRIX4RAW)
    {
        gpuCalc4xMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
        (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size());
    }

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    VECTOR3D assignVec;

    for (int i = 0; i < sizeof(double)*matrixRaw*sourceMatrices.size(); ++i)
    {
        assignVec.x = hResultMatrices[i+C1];
        assignVec.y = hResultMatrices[i+C2];
        assignVec.z = hResultMatrices[i+C3];

        resultMatrices.push_back(assignVec);
    }

    for (int i = 0; i < sizeof(double)*matrixRaw*sourceMatrices.size(); ++i)
    {
        vecSystem.pushVec3d(hResultMatrices[i+C1], hResultMatrices[i+C2], hResultMatrices[i+C3], &resultMatrices);
    }

    // Release all memory allocated by malloc
    free(hSourceMatrices);
    free(hCalcMatrices);
    free(hResultMatrices);

    cudaFree(dSourceMatrices);
    cudaFree(dCalcMatrices);
    cudaFree(dResultMatrices);
}

void MATRIX::posTrans(std::vector<VECTOR3D> sourceCoordinates, VECTOR3D change_pos_amount)
{
    matrixRaw = MATRIX4RAW;
    sourceMatrices = sourceCoordinates;
    
    VECTOR4D assignVec{0, 0, 0, 0};

    assignVec.x = 1;
    assignVec.y = 0;
    assignVec.z = 0;
    assignVec.w = 0;
    calcMatrices4x.push_back(assignVec);

    assignVec.x = 0;
    assignVec.y = 1;
    assignVec.z = 0;
    assignVec.w = 0;
    calcMatrices4x.push_back(assignVec);

    assignVec.x = 0;
    assignVec.y = 0;
    assignVec.z = 1;
    assignVec.w = 0;
    calcMatrices4x.push_back(assignVec);

    
}

void MATRIX::rotTrans
(
    std::vector<VECTOR3D> sourceCoordinates, 
    int rotationAxis, 
    double rotationAngle
)
{


    matrixRaw = MATRIX3RAW;


    switch (rotationAxis)
    {
    case SELECTAXIS_X:
        
        break;

    case SELECTAXIS_Y:
        /* code */
        break;

    case SELECTAXIS_Z:
        /* code */
        break;
    
    default:
        break;
    }
}
