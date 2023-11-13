#include "cgmath.cuh"

__global__ void gpuVecAddition
(
    double* sourceV, 
    double* calcV, 
    double* resultV, 
    int size // Number of array columns
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && j < VECTOR3)
    {
        resultV[i*VECTOR3 + j] = sourceV[i*VECTOR3 + j] + calcV[j];
    }
}

__global__ void gpuVecMinus(double *startV, double *endV, double *resultV, int size)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && j < VECTOR3)
    {
        resultV[i*VECTOR3 + j] = endV[i*VECTOR3 + j] - startV[i*VECTOR3 + j];
    }
}

__global__ void gpuVecMinusToPoint(double *startV, double *endV, double *resultV, int size)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && j < VECTOR3)
    {
        resultV[i*VECTOR3 + j] = endV[i*VECTOR3 + j] - startV[j];
    }
}

__global__ void gpuVecDotProduct
(
    double* sourceV, 
    double* calcV, 
    double* resultV, 
    int size
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        resultV[i]
        = sourceV[i*VECTOR3 + VX] * calcV[i*VECTOR3 + VX] / 
        sqrt(calcV[i*VECTOR3 + VX]*calcV[i*VECTOR3 + VX] + 
        calcV[i*VECTOR3 + VY]*calcV[i*VECTOR3 + VY] + calcV[i*VECTOR3 + VZ]*calcV[i*VECTOR3 + VZ])

        + sourceV[i*VECTOR3 + VY] * calcV[i*VECTOR3 + VY] / 
        sqrt(calcV[i*VECTOR3 + VX]*calcV[i*VECTOR3 + VX] 
        + calcV[i*VECTOR3 + VY]*calcV[i*VECTOR3 + VY] + calcV[i*VECTOR3 + VZ]*calcV[i*VECTOR3 + VZ])

        + sourceV[i*VECTOR3 + VZ] * calcV[i*VECTOR3 + VZ] / 
        sqrt(calcV[i*VECTOR3 + VX]*calcV[i*VECTOR3 + VX] 
        + calcV[i*VECTOR3 + VY]*calcV[i*VECTOR3 + VY] + calcV[i*VECTOR3 + VZ]*calcV[i*VECTOR3 + VZ]);
    }
}

__global__ void gpuVecCrossProduct
(
    double* sourceV, 
    double* calcV, 
    double* resultV, 
    int size
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        resultV[i*VECTOR3 + VX]
        = sourceV[i*VECTOR3 + VY] * calcV[i*VECTOR3 + VZ] - sourceV[i*VECTOR3 + VZ] * calcV[i*VECTOR3 + VY];

        resultV[i*VECTOR3 + VY]
        = sourceV[i*VECTOR3 + VZ] * calcV[i*VECTOR3 + VX] - sourceV[i*VECTOR3 + VX] * calcV[i*VECTOR3 + VZ];

        resultV[i*VECTOR3 + VZ]
        = sourceV[i*VECTOR3 + VX] * calcV[i*VECTOR3 + VY] - sourceV[i*VECTOR3 + VY] * calcV[i*VECTOR3 + VX];
    }
}


void VECTOR::getVec3dFromV(std::vector<VECTOR3D> startVs, std::vector<VECTOR3D> endVs)
{
    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*startVs.size());
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3*endVs.size());
    hResultVec = (double*)malloc(sizeof(double)*VECTOR3*startVs.size());

    memcpy(hSouceVec, startVs.data(), sizeof(double)*VECTOR3*startVs.size());
    memcpy(hCalcVec, endVs.data(), sizeof(double)*VECTOR3*endVs.size());

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*startVs.size());
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3*endVs.size());
    cudaMalloc((void**)&dResultVec, sizeof(double)*VECTOR3*startVs.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*startVs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3*endVs.size(), cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid
    (
        (startVs.size()*VECTOR3 + dimBlock.x - 1) / dimBlock.x, 
        (startVs.size()*VECTOR3 + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    
    gpuVecMinus<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, startVs.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*VECTOR3*startVs.size(), cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultVector3D.resize(startVs.size());
    for (int i = 0; i < startVs.size(); ++i)
    {
        inputVec3d
        (
            hResultVec[i*VECTOR3 + X0], 
            hResultVec[i*VECTOR3 + Y0], 
            hResultVec[i*VECTOR3 + Z0], 
            i,
            &resultVector3D
        );
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);
}

void VECTOR::getVec3dFaceToPolyLineStart(std::vector<CALCFACE> face, std::vector<POLYINFO> polyInfo)
{
    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    hResultVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);

    for (int i = 0; i < polyInfo.size() * POLYLINES; ++i)
    {
        for (int j = 0; j < face.size(); ++j)
        {
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = face[j].oneV.x;
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = face[j].oneV.y;
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = face[j].oneV.z;
        }
    }

    for (int i = 0; i < polyInfo.size(); ++i)
    {
        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineStartPoint[V0].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineStartPoint[V0].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineStartPoint[V0].z;
        }

        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineStartPoint[V1].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineStartPoint[V1].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineStartPoint[V1].z;
        }

        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineStartPoint[V2].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineStartPoint[V2].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineStartPoint[V2].z;
        }
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    cudaMalloc((void**)&dResultVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid
    (
        (VECTOR3*face.size() * polyInfo.size() * POLYLINES + dimBlock.x - 1) / dimBlock.x, 
        (VECTOR3*face.size() * polyInfo.size() * POLYLINES + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    
    gpuVecMinus<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, face.size() * polyInfo.size() * POLYLINES);

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultVector3D.resize(face.size() * polyInfo.size() * POLYLINES);
    for (int i = 0; i < face.size() * polyInfo.size() * POLYLINES; ++i)
    {
        inputVec3d
        (
            hResultVec[i*VECTOR3 + X0], 
            hResultVec[i*VECTOR3 + Y0], 
            hResultVec[i*VECTOR3 + Z0], 
            i,
            &resultVector3D
        );
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);
}

void VECTOR::getVec3dFaceToPolyLineEnd(std::vector<CALCFACE> face, std::vector<POLYINFO> polyInfo)
{
    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    hResultVec = (double*)malloc(sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);

    for (int i = 0; i < polyInfo.size() * POLYLINES; ++i)
    {
        for (int j = 0; j < face.size(); ++j)
        {
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = face[j].oneV.x;
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = face[j].oneV.y;
            hSouceVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = face[j].oneV.z;
        }
    }

    for (int i = 0; i < polyInfo.size(); ++i)
    {
        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineEndPoint[V0].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineEndPoint[V0].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineEndPoint[V0].z;
        }

        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineEndPoint[V1].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineEndPoint[V1].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineEndPoint[V1].z;
        }

        for (int j = 0; j < face.size(); ++j)
        {
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = polyInfo[i].lineEndPoint[V2].x;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VY] = polyInfo[i].lineEndPoint[V2].y;
            hCalcVec[i*face.size()*VECTOR3 + j*VECTOR3 + VZ] = polyInfo[i].lineEndPoint[V2].z;
        }
    }

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);
    cudaMalloc((void**)&dResultVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES);

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid
    (
        (VECTOR3*face.size() * polyInfo.size() * POLYLINES + dimBlock.x - 1) / dimBlock.x, 
        (VECTOR3*face.size() * polyInfo.size() * POLYLINES + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    
    gpuVecMinus<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, face.size() * polyInfo.size() * POLYLINES);

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*VECTOR3*face.size() * polyInfo.size() * POLYLINES, cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultVector3D.resize(face.size() * polyInfo.size() * POLYLINES);
    for (int i = 0; i < face.size() * polyInfo.size() * POLYLINES; ++i)
    {
        inputVec3d
        (
            hResultVec[i*VECTOR3 + X0], 
            hResultVec[i*VECTOR3 + Y0], 
            hResultVec[i*VECTOR3 + Z0], 
            i,
            &resultVector3D
        );
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);
}

void VECTOR::decimalLimit(VECTOR3D *v)
{
    (*v).x = std::floor(v->x * DECIMAL_PLACES) / DECIMAL_PLACES;
    (*v).y = std::floor(v->y * DECIMAL_PLACES) / DECIMAL_PLACES;
    (*v).z = std::floor(v->z * DECIMAL_PLACES) / DECIMAL_PLACES;
}

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

void VECTOR::posTrans(std::vector<VECTOR3D> sourceVec, VECTOR3D calcVec)
{
    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*sourceVec.size());
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3);
    hResultVec = (double*)malloc(sizeof(double)*VECTOR3*sourceVec.size());

    memcpy(hSouceVec, sourceVec.data(), sizeof(double)*VECTOR3*sourceVec.size());

    hCalcVec[VX] = calcVec.x;
    hCalcVec[VY] = calcVec.y;
    hCalcVec[VZ] = calcVec.z;

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*sourceVec.size());
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3);
    cudaMalloc((void**)&dResultVec, sizeof(double)*VECTOR3*sourceVec.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*sourceVec.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3, cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid
    (
        (sourceVec.size()*VECTOR3 + dimBlock.x - 1) / dimBlock.x, 
        (sourceVec.size()*VECTOR3 + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    gpuVecAddition<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, sourceVec.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*3*sourceVec.size(), cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultVector3D.resize(sourceVec.size());
    for (int i = 0; i < sourceVec.size(); ++i)
    {
        inputVec3d
        (
            hResultVec[i*VECTOR3 + VX], 
            hResultVec[i*VECTOR3 + VY], 
            hResultVec[i*VECTOR3 + VZ], 
            i,
            &resultVector3D
        );
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);
}

void VECTOR::dotProduct(std::vector<VECTOR3D> sourceVec, std::vector<VECTOR3D> calcVec)
{
    if (sourceVec.size() != calcVec.size())
    {
        OutputDebugStringA("Vector function{dotProduct} ERROR\n");
        OutputDebugStringA("souceVec and calcVec array sizes are different\n");
        return;
    }
    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*sourceVec.size());
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3*calcVec.size());
    hResultVec = (double*)malloc(sizeof(double)*sourceVec.size());

    memcpy(hSouceVec, sourceVec.data(), sizeof(double)*VECTOR3*sourceVec.size());
    memcpy(hCalcVec, calcVec.data(), sizeof(double)*VECTOR3*sourceVec.size());

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*sourceVec.size());
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3*calcVec.size());
    cudaMalloc((void**)&dResultVec, sizeof(double)*sourceVec.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*sourceVec.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3*calcVec.size(), cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    int blockSize = 1024;
    int numBlocks = (sourceVec.size() + blockSize - 1) / blockSize;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    gpuVecDotProduct<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, sourceVec.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*sourceVec.size(), cudaMemcpyDeviceToHost);


    // Assign the result to a Vector member variable
    resultVector.resize(sourceVec.size());
    for (int i = 0; i < sourceVec.size(); ++i)
    {
        resultVector[i] = hResultVec[i];
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);

}

void VECTOR::crossProduct(std::vector<VECTOR3D> sourceVec, std::vector<VECTOR3D> calcVec)
{
    if (sourceVec.size() != calcVec.size())
    {
        OutputDebugStringA("Vector function{crossProduct} ERROR\n");
        OutputDebugStringA("souceVec and calcVec array sizes are different\n");
        return;
    }

    // Allocate memory for each vector size
    hSouceVec = (double*)malloc(sizeof(double)*VECTOR3*sourceVec.size());
    hCalcVec = (double*)malloc(sizeof(double)*VECTOR3*calcVec.size());
    hResultVec = (double*)malloc(sizeof(double)*VECTOR3*sourceVec.size());

    memcpy(hSouceVec, sourceVec.data(), sizeof(double)*3*sourceVec.size());
    memcpy(hCalcVec, calcVec.data(), sizeof(double)*3*calcVec.size());

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dSouceVec, sizeof(double)*VECTOR3*sourceVec.size());
    cudaMalloc((void**)&dCalcVec, sizeof(double)*VECTOR3*calcVec.size());
    cudaMalloc((void**)&dResultVec, sizeof(double)*VECTOR3*sourceVec.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dSouceVec, hSouceVec, sizeof(double)*VECTOR3*sourceVec.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcVec, hCalcVec, sizeof(double)*VECTOR3*calcVec.size(), cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    int blockSize = 1024;
    int numBlocks = (sourceVec.size()*VECTOR3 + blockSize - 1) / blockSize;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    (
        (sourceVec.size()*VECTOR3 + dimBlock.x - 1) / dimBlock.x, 
        (sourceVec.size()*VECTOR3 + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    gpuVecCrossProduct<<<dimGrid, dimBlock>>>
    (dSouceVec, dCalcVec, dResultVec, sourceVec.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultVec, dResultVec, sizeof(double)*VECTOR3*sourceVec.size(), cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultVector3D.resize(sourceVec.size());
    for (int i = 0; i < sourceVec.size(); ++i)
    {
        inputVec3d
        (
            hResultVec[i*VECTOR3 + VX], 
            hResultVec[i*VECTOR3 + VY], 
            hResultVec[i*VECTOR3 + VZ], 
            i,
            &resultVector3D
        );
    }

    // Release all memory allocated by malloc
    free(hSouceVec);
    free(hCalcVec);
    free(hResultVec);

    cudaFree(dSouceVec);
    cudaFree(dCalcVec);
    cudaFree(dResultVec);
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
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((sourceMatrices.size() + dimBlock.x - 1) 
    / dimBlock.x, (sourceMatrices.size() + dimBlock.y - 1) / dimBlock.y); // Grid Size
    gpuCalc3xMatrixProduct<<<dimGrid, dimBlock>>>
    (dSourceMatrices, dCalcMatrices, dResultMatrices, sourceMatrices.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultMatrices, dResultMatrices, sizeof(double)*MATRIX3RAW*sourceMatrices.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    resultMatrices.resize(sourceMatrices.size());
    for (int i = 0; i < sourceMatrices.size(); ++i)
    {
        vec.inputVec3d
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

__global__ void gpuGetLinePlaneI
(
    double* lineVA,
    double* lineVB,
    double* dotPaN,
    double* dotPbN,
    double* lpI,
    int amoutI
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < amoutI && j < VECTOR3)
    {
        lpI[i*VECTOR3 + j] = lineVA[i*VECTOR3 + j] + 
        (lineVB[i*VECTOR3 + j] - lineVA[i*VECTOR3 + j]) * (abs(dotPaN[i]) / (abs(dotPaN[i]) + abs(dotPbN[i])));
    } 
}

void EQUATION::getLinePlaneI
(
    std::vector<VECTOR3D> lineVA,
    std::vector<VECTOR3D> lineVB,
    std::vector<VECTOR3D> planeV,
    std::vector<VECTOR3D> planeN
)
{
    std::vector<VECTOR3D> planeVperLine;
    planeVperLine.resize(planeV.size() * lineVA.size());
    for (int i = 0; i < lineVA.size(); ++i)
    {
        for (int j = 0; j < planeV.size(); ++j)
        {
            planeVperLine[i*planeV.size() + j] = planeV[j];
        }
    }

    std::vector<VECTOR3D> lineVAperPlane;
    lineVAperPlane.resize(planeV.size() * lineVA.size());
    for (int i = 0; i < lineVA.size(); ++i)
    {
        for (int j = 0; j < planeV.size(); ++j)
        {
            lineVAperPlane[i*planeV.size() + j] = lineVA[i];
        }
    }

    std::vector<VECTOR3D> lineVBperPlne;
    lineVBperPlne.resize(planeV.size() * lineVB.size());
    for (int i = 0; i < lineVB.size(); ++i)
    {
        for (int j = 0; j < planeV.size(); ++j)
        {
            lineVBperPlne[i*planeV.size() + j] = lineVB[i];
        }
    }

    // Find a vector from a point on the surface to a polygon vertex
    vec.getVec3dFromV(planeVperLine, lineVAperPlane);
    vPvLa = vec.resultVector3D;

    vec.getVec3dFromV(planeVperLine, lineVBperPlne);
    vPvLb = vec.resultVector3D;

    std::vector<VECTOR3D> planeNperLine;
    planeNperLine.resize(planeN.size() * lineVA.size());
    for (int i = 0; i < lineVA.size(); ++i)
    {
        for (int j = 0; j < planeN.size(); ++j)
        {
            planeNperLine[i*planeN.size() + j] = planeN[j];
        }
    }

    vec.dotProduct(vPvLa, planeNperLine);
    pnDotPvLa = vec.resultVector;

    vec.dotProduct(vPvLb, planeNperLine);
    pnDotPvLb = vec.resultVector;

    existenceI.resize(0);
    existenceI.resize(lineVA.size());

    std::vector<VECTOR3D> calcLineVA;
    std::vector<VECTOR3D> calcLineVB;
    std::vector<double> calcPnDotPvLa;
    std::vector<double> calcPnDotPvLb;

    bool findI = false;
    for (int i = 0; i < lineVA.size(); ++i)
    {
        existenceI[i].resize(planeN.size(), I_FALSE);
        for (int j = 0; j < planeN.size(); ++j)
        {
            if (pnDotPvLa[i*planeN.size() + j] >= 0)
            {
                if (pnDotPvLb[i*planeN.size() + j] <= 0)
                {
                    existenceI[i][j] = I_TRUE;
                    findI = true;
                    calcLineVA.push_back(lineVA[i]);
                    calcLineVB.push_back(lineVB[i]);
                    calcPnDotPvLa.push_back(pnDotPvLa[i*planeN.size() + j]);
                    calcPnDotPvLb.push_back(pnDotPvLb[i*planeN.size() + j]);
                }
            }

            if (pnDotPvLa[i*planeN.size() + j] <= 0 && !findI)
            {
                if (pnDotPvLb[i*planeN.size() + j] >= 0)
                {
                    existenceI[i][j] = I_TRUE;
                    calcLineVA.push_back(lineVA[i]);
                    calcLineVB.push_back(lineVB[i]);
                    calcPnDotPvLa.push_back(pnDotPvLa[i*planeN.size() + j]);
                    calcPnDotPvLb.push_back(pnDotPvLb[i*planeN.size() + j]);
                }
            }
            findI = false;
        }
    }

    int amoutI = calcLineVA.size();
    // Allocate memory for each matrix size
    hCalcLineVA = (double*)malloc(sizeof(double)*VECTOR3*amoutI);
    hCalcLineVB = (double*)malloc(sizeof(double)*VECTOR3*amoutI);
    hCalcPnDotPvLa = (double*)malloc(sizeof(double)*amoutI);
    hCalcPnDotPvLb = (double*)malloc(sizeof(double)*amoutI);
    hLinePlaneI = (double*)malloc(sizeof(double)*VECTOR3*amoutI);

    // Copy member variable
    memcpy(hCalcLineVA, calcLineVA.data(), sizeof(double)*VECTOR3*amoutI);
    memcpy(hCalcLineVB, calcLineVB.data(), sizeof(double)*VECTOR3*amoutI);
    memcpy(hCalcPnDotPvLa, calcPnDotPvLa.data(), sizeof(double)*amoutI);
    memcpy(hCalcPnDotPvLb, calcPnDotPvLb.data(), sizeof(double)*amoutI);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dCalcLineVA, sizeof(double)*VECTOR3*amoutI);
    cudaMalloc((void**)&dCalcLineVB, sizeof(double)*VECTOR3*amoutI);
    cudaMalloc((void**)&dCalcPnDotPvLa, sizeof(double)*amoutI);
    cudaMalloc((void**)&dCalcPnDotPvLb, sizeof(double)*amoutI);
    cudaMalloc((void**)&dLinePlaneI, sizeof(double)*VECTOR3*amoutI);

    // Copy host-side data to device-side memory
    cudaMemcpy(dCalcLineVA, hCalcLineVA, sizeof(double)*VECTOR3*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcLineVB, hCalcLineVB, sizeof(double)*VECTOR3*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcPnDotPvLa, hCalcPnDotPvLa, sizeof(double)*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcPnDotPvLb, hCalcPnDotPvLb, sizeof(double)*amoutI, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((VECTOR3 + dimBlock.x - 1) 
    / dimBlock.x, (amoutI + dimBlock.y - 1) / dimBlock.y); // Grid Size
    gpuGetLinePlaneI<<<dimGrid, dimBlock>>>
    (dCalcLineVA, dCalcLineVB, dCalcPnDotPvLa, dCalcPnDotPvLb, dLinePlaneI, amoutI);

    // Copy results from device memory to host memory
    cudaMemcpy(hLinePlaneI, dLinePlaneI, sizeof(double)*VECTOR3*amoutI, cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    linePlaneI.resize(amoutI);
    for (int i = 0; i < amoutI; ++i)
    {
        vec.inputVec3d
        (
            hLinePlaneI[i*VECTOR3+X0], 
            hLinePlaneI[i*VECTOR3+Y0], 
            hLinePlaneI[i*VECTOR3+Z0], 
            i,
            &linePlaneI
        );
    }

    // Release all memory allocated by malloc
    free(hCalcLineVA);
    free(hCalcLineVB);
    free(hCalcPnDotPvLa);
    free(hCalcPnDotPvLb);
    free(hLinePlaneI);

    cudaFree(dCalcLineVA);
    cudaFree(dCalcLineVB);
    cudaFree(dCalcPnDotPvLa);
    cudaFree(dCalcPnDotPvLb);
    cudaFree(dLinePlaneI);
}

void EQUATION::getLinePlaneI(std::vector<POLYINFO> polyInfo, std::vector<CALCFACE> face)
{
    vec.getVec3dFaceToPolyLineStart(face, polyInfo);
    std::vector<VECTOR3D> vecPlaneVLineStart = vec.resultVector3D;

    vec.getVec3dFaceToPolyLineEnd(face, polyInfo);
    std::vector<VECTOR3D> vecPlaneVLineEnd = vec.resultVector3D;

    std::vector<VECTOR3D> planeNperLine;
    planeNperLine.resize(polyInfo.size() * POLYLINES * face.size());
    for (int i = 0; i < polyInfo.size() * POLYLINES; ++i)
    {
        for (int j = 0; j < face.size(); ++j)
        {
            planeNperLine[i*face.size()*VECTOR3 + j*VECTOR3 + VX] = face[j].normal;
        }
    }

    vec.dotProduct(vecPlaneVLineStart, planeNperLine);
    std::vector<double> pnDotPlaneVLineStart = vec.resultVector;

    vec.dotProduct(vecPlaneVLineEnd, planeNperLine);
    std::vector<double> pnDotPlaneVLineEnd = vec.resultVector;

    existenceI.resize(0);
    existenceI.resize(lineVA.size());

    std::vector<VECTOR3D> calcLineVA;
    std::vector<VECTOR3D> calcLineVB;
    std::vector<double> calcPnDotPvLa;
    std::vector<double> calcPnDotPvLb;

    bool findI = false;
    for (int i = 0; i < lineVA.size(); ++i)
    {
        existenceI[i].resize(planeN.size(), I_FALSE);
        for (int j = 0; j < planeN.size(); ++j)
        {
            if (pnDotPvLa[i*planeN.size() + j] >= 0)
            {
                if (pnDotPvLb[i*planeN.size() + j] <= 0)
                {
                    existenceI[i][j] = I_TRUE;
                    findI = true;
                    calcLineVA.push_back(lineVA[i]);
                    calcLineVB.push_back(lineVB[i]);
                    calcPnDotPvLa.push_back(pnDotPvLa[i*planeN.size() + j]);
                    calcPnDotPvLb.push_back(pnDotPvLb[i*planeN.size() + j]);
                }
            }

            if (pnDotPvLa[i*planeN.size() + j] <= 0 && !findI)
            {
                if (pnDotPvLb[i*planeN.size() + j] >= 0)
                {
                    existenceI[i][j] = I_TRUE;
                    calcLineVA.push_back(lineVA[i]);
                    calcLineVB.push_back(lineVB[i]);
                    calcPnDotPvLa.push_back(pnDotPvLa[i*planeN.size() + j]);
                    calcPnDotPvLb.push_back(pnDotPvLb[i*planeN.size() + j]);
                }
            }
            findI = false;
        }
    }

    int amoutI = calcLineVA.size();
    // Allocate memory for each matrix size
    hCalcLineVA = (double*)malloc(sizeof(double)*VECTOR3*amoutI);
    hCalcLineVB = (double*)malloc(sizeof(double)*VECTOR3*amoutI);
    hCalcPnDotPvLa = (double*)malloc(sizeof(double)*amoutI);
    hCalcPnDotPvLb = (double*)malloc(sizeof(double)*amoutI);
    hLinePlaneI = (double*)malloc(sizeof(double)*VECTOR3*amoutI);

    // Copy member variable
    memcpy(hCalcLineVA, calcLineVA.data(), sizeof(double)*VECTOR3*amoutI);
    memcpy(hCalcLineVB, calcLineVB.data(), sizeof(double)*VECTOR3*amoutI);
    memcpy(hCalcPnDotPvLa, calcPnDotPvLa.data(), sizeof(double)*amoutI);
    memcpy(hCalcPnDotPvLb, calcPnDotPvLb.data(), sizeof(double)*amoutI);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dCalcLineVA, sizeof(double)*VECTOR3*amoutI);
    cudaMalloc((void**)&dCalcLineVB, sizeof(double)*VECTOR3*amoutI);
    cudaMalloc((void**)&dCalcPnDotPvLa, sizeof(double)*amoutI);
    cudaMalloc((void**)&dCalcPnDotPvLb, sizeof(double)*amoutI);
    cudaMalloc((void**)&dLinePlaneI, sizeof(double)*VECTOR3*amoutI);

    // Copy host-side data to device-side memory
    cudaMemcpy(dCalcLineVA, hCalcLineVA, sizeof(double)*VECTOR3*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcLineVB, hCalcLineVB, sizeof(double)*VECTOR3*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcPnDotPvLa, hCalcPnDotPvLa, sizeof(double)*amoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dCalcPnDotPvLb, hCalcPnDotPvLb, sizeof(double)*amoutI, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((VECTOR3 + dimBlock.x - 1) 
    / dimBlock.x, (amoutI + dimBlock.y - 1) / dimBlock.y); // Grid Size
    gpuGetLinePlaneI<<<dimGrid, dimBlock>>>
    (dCalcLineVA, dCalcLineVB, dCalcPnDotPvLa, dCalcPnDotPvLb, dLinePlaneI, amoutI);

    // Copy results from device memory to host memory
    cudaMemcpy(hLinePlaneI, dLinePlaneI, sizeof(double)*VECTOR3*amoutI, cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    linePlaneI.resize(amoutI);
    for (int i = 0; i < amoutI; ++i)
    {
        vec.inputVec3d
        (
            hLinePlaneI[i*VECTOR3+X0], 
            hLinePlaneI[i*VECTOR3+Y0], 
            hLinePlaneI[i*VECTOR3+Z0], 
            i,
            &linePlaneI
        );
    }

    // Release all memory allocated by malloc
    free(hCalcLineVA);
    free(hCalcLineVB);
    free(hCalcPnDotPvLa);
    free(hCalcPnDotPvLb);
    free(hLinePlaneI);

    cudaFree(dCalcLineVA);
    cudaFree(dCalcLineVB);
    cudaFree(dCalcPnDotPvLa);
    cudaFree(dCalcPnDotPvLb);
    cudaFree(dLinePlaneI);
}

__global__ void gpuGet2dAngle(double *vertValue, double *horizValue, double* resultValue, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        resultValue[i] = atan2(vertValue[i], horizValue[i]) * 180 / PI;
    }
}

void TRIANGLE_RATIO::get2dVecAngle(std::vector<double> vertValue, std::vector<double> horizValue)
{
    // Allocate memory for each vector size
    hHorizValue = (double*)malloc(sizeof(double)*horizValue.size());
    hVertValue = (double*)malloc(sizeof(double)*vertValue.size());
    hResultValue = (double*)malloc(sizeof(double)*horizValue.size());

    memcpy(hHorizValue, horizValue.data(), sizeof(double)*horizValue.size());
    memcpy(hVertValue, vertValue.data(), sizeof(double)*vertValue.size());

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dHorizValue, sizeof(double)*VECTOR3*horizValue.size());
    cudaMalloc((void**)&dVertValue, sizeof(double)*VECTOR3*vertValue.size());
    cudaMalloc((void**)&dResultValue, sizeof(double)*VECTOR3*horizValue.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dHorizValue, hHorizValue, sizeof(double)*horizValue.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dVertValue, hVertValue, sizeof(double)*vertValue.size(), cudaMemcpyHostToDevice);
    
    // GPU kernel function calls
    int blockSize = 1024;
    int numBlocks = (horizValue.size()*VECTOR3 + blockSize - 1) / blockSize;
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    (
        (horizValue.size()*VECTOR3 + dimBlock.x - 1) / dimBlock.x, 
        (horizValue.size()*VECTOR3 + dimBlock.y - 1) / dimBlock.y
    ); // Grid Size
    gpuGet2dAngle<<<dimGrid, dimBlock>>>
    (dVertValue, dHorizValue, dResultValue, horizValue.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hResultValue, dResultValue, sizeof(double)*horizValue.size(), cudaMemcpyDeviceToHost);

    // Assign the result to a Vector member variable
    resultDegree.resize(horizValue.size());
    for (int i = 0; i < horizValue.size(); ++i)
    {
        resultDegree[i] = hResultValue[i];
    }

    // Release all memory allocated by malloc
    free(hHorizValue);
    free(hVertValue);
    free(hResultValue);

    cudaFree(dHorizValue);
    cudaFree(dVertValue);
    cudaFree(dResultValue);
}

