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

    memcpy(hSouceVec, sourceVec.data(), sizeof(double)*3*sourceVec.size());

    for (int i = 0; i < calcVec.size(); ++i)
    {
        hCalcVec[i*VECTOR3 + 0] = calcVec[i].x;
        hCalcVec[i*VECTOR3 + 1] = calcVec[i].y;
        hCalcVec[i*VECTOR3 + 2] = calcVec[i].z;
    }

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

__global__ void gpuGetDenominateT
(
    double* lineVB, // l, m, n
    double* planeN, // p, q, r
    double* t,
    int lineAmout,
    int planeAmout
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < lineAmout && j < planeAmout)
    {
        t[i*planeAmout + j]
        = planeN[i*6 + j*VECTOR3 + X0] * lineVB[i*VECTOR3 + X0]
        + planeN[i*6 + j*VECTOR3 + Y0] * lineVB[i*VECTOR3 + Y0]
        + planeN[i*6 + j*VECTOR3 + Z0] * lineVB[i*VECTOR3 + Z0];
    }
}

__global__ void gpuGetLinePlaneI
(
    double* lineVA,
    double* lineVB,
    double* t,
    double* lpI,
    int amoutI
)
{
    // Decide which (i,j) you are in charge of based on your back number
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < amoutI && j < VECTOR3)
    {
        lpI[i*VECTOR3 + j] = 
        lineVA[i*VECTOR3 + j] + t[i] * lineVB[i*VECTOR3 + j];
    }
}

void EQUATION::getDenominateT(std::vector<VECTOR3D> lineVB, std::vector<VECTOR3D> planeN)
{
    // Allocate memory for each matrix size
    hLineVertexB = (double*)malloc(sizeof(double)*VECTOR3*lineVB.size());
    hPlaneNormal = (double*)malloc(sizeof(double)*VECTOR3*planeN.size());
    hParaT = (double*)malloc(sizeof(double)*lineVB.size()*planeN.size());

    // Copy member variable
    memcpy(hLineVertexB, lineVB.data(), sizeof(double)*VECTOR3*lineVB.size());
    memcpy(hPlaneNormal, planeN.data(), sizeof(double)*VECTOR3*planeN.size());

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dLineVertexB, sizeof(double)*VECTOR3*lineVB.size());
    cudaMalloc((void**)&dPlaneNormal, sizeof(double)*VECTOR3*planeN.size());
    cudaMalloc((void**)&dParaT, sizeof(double)*lineVB.size()*planeN.size());

    // Copy host-side data to device-side memory
    cudaMemcpy(dLineVertexB, hLineVertexB, sizeof(double)*VECTOR3*lineVB.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dPlaneNormal, hPlaneNormal, sizeof(double)*VECTOR3*planeN.size(), cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((lineVB.size() + dimBlock.x - 1) / dimBlock.x, 
    (planeN.size() + dimBlock.y - 1) / dimBlock.y); // Grid Size
    gpuGetDenominateT<<<dimGrid, dimBlock>>>
    (dLineVertexB, dPlaneNormal, dParaT, lineVB.size(), planeN.size());

    // Copy results from device memory to host memory
    cudaMemcpy(hParaT, dParaT, sizeof(double)*lineVB.size()*planeN.size(), cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    paraT.resize(lineVB.size()*planeN.size());
    for (int i = 0; i < lineVB.size()*planeN.size(); ++i)
    {
        paraT[i] = hParaT[i];
    }

    // Release all memory allocated by malloc
    free(hLineVertexA);
    free(hLineVertexB);
    free(hPlaneVertex);
    free(hPlaneNormal);
    free(hLinePlaneI);

    cudaFree(dLineVertexA);
    cudaFree(dLineVertexB);
    cudaFree(dPlaneVertex);
    cudaFree(dPlaneNormal);
    cudaFree(dLinePlaneI);
}

void EQUATION::getLinePlaneI
(
    std::vector<VECTOR3D> lineVA,
    std::vector<VECTOR3D> lineVB,
    std::vector<VECTOR3D> planeV,
    std::vector<VECTOR3D> planeN
)
{
    // Obtaining the value of the denominator of the mediating variable t
    getDenominateT(lineVB, planeN);

    // Checking the value of the mediating variable t to avoid division by zero
    int sumAmoutI = 0;
    amoutIeachLine.resize(0);
    std::vector<VECTOR3D> calcLineVA;
    std::vector<VECTOR3D> calcLineVB;
    std::vector<double> calcParaT;

    std::vector<INT2D> iPerLine;
    iPerLine.resize(1);
    iPerLine[0].n.resize(planeN.size());

    for (int i = 0; i < lineVA.size(); ++i)
    {
        for (int j = 0; j < planeN.size(); ++j)
        {
            if (paraT[i] != 0)
            {
                calcLineVA.push_back(lineVA[i]);
                calcLineVB.push_back(planeV[j]);
                calcParaT.push_back(paraT[i]);
                iPerLine[0].n[j] = I_TRUE;
                sumAmoutI += 1;
            }
        }
        amoutIeachLine.push_back(iPerLine[0]);
        
        for (int j = 0; j < planeN.size(); ++j)
        {
            iPerLine[0].n[j] = I_FALSE;
        }
    }

    // Allocate memory for each matrix size
    hLineVertexA = (double*)malloc(sizeof(double)*VECTOR3*sumAmoutI);
    hLineVertexB = (double*)malloc(sizeof(double)*VECTOR3*sumAmoutI);
    hParaT = (double*)malloc(sizeof(double)*sumAmoutI);
    hLinePlaneI = (double*)malloc(sizeof(double)*VECTOR3*sumAmoutI);

    // Copy member variable
    memcpy(hLineVertexA, calcLineVA.data(), sizeof(double)*VECTOR3*sumAmoutI);
    memcpy(hLineVertexB, calcLineVA.data(), sizeof(double)*VECTOR3*sumAmoutI);
    memcpy(hParaT, calcParaT.data(), sizeof(double)*sumAmoutI);

    // Allocate device-side memory using CUDAMALLOC
    cudaMalloc((void**)&dLineVertexA, sizeof(double)*VECTOR3*sumAmoutI);
    cudaMalloc((void**)&dLineVertexB, sizeof(double)*VECTOR3*sumAmoutI);
    cudaMalloc((void**)&dParaT, sizeof(double)*sumAmoutI);
    cudaMalloc((void**)&dLinePlaneI, sizeof(double)*VECTOR3*sumAmoutI);

    // Copy host-side data to device-side memory
    cudaMemcpy(dLineVertexA, hLineVertexA, sizeof(double)*VECTOR3*sumAmoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dLineVertexB, hLineVertexB, sizeof(double)*VECTOR3*sumAmoutI, cudaMemcpyHostToDevice);
    cudaMemcpy(dParaT, hParaT, sizeof(double)*sumAmoutI, cudaMemcpyHostToDevice);

    // GPU kernel function calls
    dim3 dimBlock(32, 32); // Thread block size
    dim3 dimGrid((sumAmoutI + dimBlock.x - 1) / dimBlock.x, 
    (VECTOR3 + dimBlock.y - 1) / dimBlock.y); // Grid Size
    gpuGetLinePlaneI<<<dimGrid, dimBlock>>>
    (dLineVertexA, dLineVertexB, dParaT, dLinePlaneI, sumAmoutI);

    // Copy results from device memory to host memory
    cudaMemcpy(hLinePlaneI, dLinePlaneI, sizeof(double)*VECTOR3*sumAmoutI, cudaMemcpyDeviceToHost);
    
    // Assign the result to a Vector member variable
    linePlaneI.resize(sumAmoutI);
    for (int i = 0; i < sumAmoutI; ++i)
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
    free(hLineVertexA);
    free(hPlaneVertex);
    free(hPlaneNormal);
    free(hParaT);
    free(hLinePlaneI);

    cudaFree(dLineVertexA);
    cudaFree(dPlaneVertex);
    cudaFree(dPlaneNormal);
    cudaFree(dParaT);
    cudaFree(dLinePlaneI);
}

