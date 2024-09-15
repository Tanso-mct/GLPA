#ifndef GLPA_SORT_CU_H_
#define GLPA_SORT_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Constant.h"

namespace Glpa
{

typedef struct _PAIR
{
    float val1, val2;

    __device__ __host__ _PAIR()
    {
        val1 = 0;
        val2 = 0;
    }

    __device__ __host__ _PAIR(int v1, int v2)
    {
        val1 = v1;
        val2 = v2;
    }
} PAIR;


typedef struct _LIST_3
{
    int size;
    Glpa::PAIR pair[3];

    __device__ __host__ _LIST_3()
    {
        size = 0;
    }

    __device__ __host__ void push(Glpa::PAIR p)
    {
        if (size < 3)
        {
            pair[size] = p;
            size++;
        }
    }

    __device__ __host__ void aSortByVal1()
    {
        for (int i = 0; i < size-1; i++)
        {
            for (int j = 0; j < size-1; j++)
            {
                GPU_IF(pair[j].val1 > pair[j+1].val1, br4)
                {
                    Glpa::PAIR temp = pair[j];
                    pair[j] = pair[j + 1];
                    pair[j + 1] = temp;
                }
            }
        }    
    }

    __device__ __host__ void aSortByVal2()
    {
        for (int i = 0; i < size-1; i++)
        {
            for (int j = 0; j < size-1; j++)
            {
                GPU_IF(pair[j].val2 > pair[j+1].val2, br4)
                {
                    Glpa::PAIR temp = pair[j];
                    pair[j] = pair[j + 1];
                    pair[j + 1] = temp;
                }
            }
        }    
    }

    __device__ __host__ void dSortByVal1()
    {
        for (int i = 0; i < size-1; i++)
        {
            for (int j = 0; j < size-1; j++)
            {
                GPU_IF(pair[j].val1 < pair[j+1].val1, br4)
                {
                    Glpa::PAIR temp = pair[j];
                    pair[j] = pair[j + 1];
                    pair[j + 1] = temp;
                }
            }
        }    
    }

    __device__ __host__ void dSortByVal2()
    {
        for (int i = 0; i < size-1; i++)
        {
            for (int j = 0; j < size-1; j++)
            {
                GPU_IF(pair[j].val2 < pair[j+1].val2, br4)
                {
                    Glpa::PAIR temp = pair[j];
                    pair[j] = pair[j + 1];
                    pair[j + 1] = temp;
                }
            }
        }    
    }


} LIST3;



}



#endif GLPA_SORT_CU_H_