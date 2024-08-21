#include "Matrix.cuh"
#include "ErrorHandler.h"

Glpa::Matrix::Matrix(std::initializer_list<float> list)
{
    set(list);
}

Glpa::Matrix::~Matrix()
{
}

void Glpa::Matrix::set(std::initializer_list<float> list)
{
    if (list.size() == 16)
    {
        auto it = list.begin();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m4x4[i][j] = *it;
                it++;
            }
        }
    }
    else if (list.size() == 9)
    {
        auto it = list.begin();
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                m3x3[i][j] = *it;
                it++;
            }
        }
    }
    else
    {
        Glpa::runTimeError(__FILE__, __LINE__, "Matrix::set() : invalid argument size");
    }
    
}

void Glpa::Matrix::put(float matrix[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix[i][j] = m3x3[i][j];
        }
    }
}

void Glpa::Matrix::put(float matrix[4][4])
{
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix[i][j] = m4x4[i][j];
        }
    }
}
