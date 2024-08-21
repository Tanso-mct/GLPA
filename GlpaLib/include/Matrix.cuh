#ifndef GLPA_MATRIX_H_
#define GLPA_MATRIX_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <initializer_list>

#include "Vector.cuh"

namespace Glpa
{

typedef struct _GPU_MAT_4X4
{
    float m[4][4] = {0};

    __device__ __host__ _GPU_MAT_4X4() {}

    __device__ __host__ _GPU_MAT_4X4(float list[16])
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = list[i * 4 + j];
            }
        }
    }

    __device__ __host__ void set(float list[16])
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m[i][j] = list[i * 4 + j];
            }
        }
    }

    __device__ __host__ Glpa::GPU_VEC_3D productLeft3x1(Glpa::GPU_VEC_3D& vec)
    {
        Glpa::GPU_VEC_3D result;
        result.x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z + m[0][3];
        result.y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z + m[1][3];
        result.z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z + m[2][3];

        return result;
    }

} GPU_MAT_4X4;
    
class Matrix
{
public :
    float m3x3[3][3] = {0};
    float m4x4[4][4] = {0};

    Matrix(std::initializer_list<float> list);
    ~Matrix();

    void set(std::initializer_list<float> list);

    void put(float matrix[3][3]);
    void put(float matrix[4][4]);
};

} // namespace Glpa


#endif GLPA_MATRIX_H_

