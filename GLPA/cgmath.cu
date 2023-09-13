#include "cgmath.cuh"

__global__ void MATRIX::gpuCalcMatrixProduct
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

    if (i < n && j < 3)
    {
        resultMatrices[matrixRaw*i+j] 
        = calcMatrices[j] * sourceMatrices[i].x 
        + calcMatrices[j+3] * sourceMatrices[i].y 
        + calcMatrices[j+6] * sourceMatrices[i].z;
    }

}

void MATRIX::calcMatrixProduct()
{
    // Allocate memory for each matrix size
    hSourceMatrices = (VECTOR3D *)malloc(sizeof(VECTOR3D)*sourceMatrices.size());
    hCalcMatrices = (double *)malloc(sizeof(double)*matrixRaw*calcMatrices.size());
    hResultMatrices = (double *)calloc(matrixRaw*sourceMatrices.size(), sizeof(double));

    // Copy member variable
    hSourceMatrices = sourceMatrices.data();

    for (int i = 0; i < calcMatrices.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            hCalcMatrices[i+j] = calcMatrices[i].x;
            hCalcMatrices[i+j] = calcMatrices[i].y;
            hCalcMatrices[i+j] = calcMatrices[i].z;
        }
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size());
    cudaMalloc((void**)&dCalcMatrices, sizeof(double)*matrixRaw*calcMatrices.size());
    cudaMalloc((void**)&dResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSourceMatrices, hSourceMatrices, sizeof(VECTOR3D)*sourceMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcMatrices, hCalcMatrices, sizeof(double)*matrixRaw*calcMatrices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dResultMatrices, hResultMatrices, sizeof(double)*matrixRaw*sourceMatrices.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    gpuCalcMatrixProduct<<<dim3((sourceMatrices.size()+BS-1)/BS, ((sourceMatrices.size()+BS-1)/BS)), dim3(BS, BS)>>>
    (hSourceMatrices, hCalcMatrices, hResultMatrices, sourceMatrices.size());

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

    // Release all memory allocated by malloc
    free(hSourceMatrices);
    free(hCalcMatrices);
    free(hResultMatrices);

    cudaFree(dSourceMatrices);
    cudaFree(dCalcMatrices);
    cudaFree(dResultMatrices);
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
