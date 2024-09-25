#ifndef GLPA_GPU_DATA_CU_H_
#define GLPA_GPU_DATA_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Vector.cuh"

#include "Constant.h"

namespace Glpa
{

typedef struct _GPU_PAIR
{
    float val1, val2;

    __device__ __host__ _GPU_PAIR()
    {
        val1 = 0;
        val2 = 0;
    }

    __device__ __host__ _GPU_PAIR(int v1, int v2)
    {
        val1 = v1;
        val2 = v2;
    }
} GPU_PAIR;


typedef struct _GPU_LIST_3
{
    int size;
    Glpa::GPU_PAIR pair[3];

    __device__ __host__ _GPU_LIST_3()
    {
        size = 0;
    }

    __device__ __host__ void push(Glpa::GPU_PAIR p)
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
                GPU_IF(pair[j].val1 < pair[j+1].val1, br4)
                {
                    Glpa::GPU_PAIR temp = pair[j];
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
                GPU_IF(pair[j].val2 < pair[j+1].val2, br4)
                {
                    Glpa::GPU_PAIR temp = pair[j];
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
                GPU_IF(pair[j].val1 > pair[j+1].val1, br4)
                {
                    Glpa::GPU_PAIR temp = pair[j];
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
                GPU_IF(pair[j].val2 > pair[j+1].val2, br4)
                {
                    Glpa::GPU_PAIR temp = pair[j];
                    pair[j] = pair[j + 1];
                    pair[j + 1] = temp;
                }
            }
        }    
    }


} GPU_LIST3;

typedef struct _GPU_ARRAY_VEC_3D GPU_ARRAY_VEC_3D;

typedef struct _GPU_ARRAY_VEC_3D
{
    _GPU_ARRAY_VEC_3D* next = nullptr;
    Glpa::GPU_VEC_3D val;
} GPU_ARRAY_VEC_3D;

typedef struct _GPU_ARRAY_
{
    int size;
    Glpa::GPU_ARRAY_VEC_3D* head;

    __device__ __host__ _GPU_ARRAY_()
    {
        size = 0;
        head = nullptr;
    }

    __device__ __host__ void push(Glpa::GPU_VEC_3D val)
    {
        Glpa::GPU_ARRAY_VEC_3D* newVal = new Glpa::GPU_ARRAY_VEC_3D;
        newVal->val = val;
        newVal->next = nullptr;

        if (head == nullptr)
        {
            head = newVal;
        }
        else
        {
            Glpa::GPU_ARRAY_VEC_3D* temp = head;
            while (temp->next != nullptr)
            {
                temp = temp->next;
            }
            temp->next = newVal;
        }
        size++;
    }

    __device__ __host__ void pop()
    {
        if (head != nullptr)
        {
            Glpa::GPU_ARRAY_VEC_3D* temp = head;
            head = head->next;
            delete temp;
            size--;
        }
    }

    __device__ __host__ void clear()
    {
        while (head != nullptr)
        {
            Glpa::GPU_ARRAY_VEC_3D* temp = head;
            head = head->next;
            delete temp;
        }
        size = 0;
    }

    __device__ __host__ Glpa::GPU_VEC_3D* get(int index)
    {
        if (index < size)
        {
            Glpa::GPU_ARRAY_VEC_3D* temp = head;
            for (int i = 0; i < index; i++)
            {
                temp = temp->next;
            }
            return &temp->val;
        }
        else
        {
            return nullptr;
        }
    }

    __device__ __host__ void set(int index, Glpa::GPU_VEC_3D val)
    {
        if (index < size)
        {
            Glpa::GPU_ARRAY_VEC_3D* temp = head;
            for (int i = 0; i < index; i++)
            {
                temp = temp->next;
            }
            temp->val = val;
        }
    }

} GPU_ARRAY;






}



#endif GLPA_GPU_DATA_CU_H_